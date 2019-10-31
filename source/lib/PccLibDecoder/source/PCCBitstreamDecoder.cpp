/* The copyright in this software is being made available under the BSD
 * License, included below. This software may be subject to other third party
 * and contributor rights, including patent rights, and no such rights are
 * granted under this license.
 *
 * Copyright (c) 2010-2017, ISO/IEC
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *  * Neither the name of the ISO/IEC nor the names of its contributors may
 *    be used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 */
#include "PCCCommon.h"
#include "PCCVideoBitstream.h"
#include "PCCBitstream.h"
#include "PCCContext.h"
#include "PCCFrameContext.h"
#include "PCCPatch.h"

#include "PCCBitstreamDecoder.h"

using namespace pcc;

PCCBitstreamDecoder::PCCBitstreamDecoder() : prevPatchSizeU_( 0 ), prevPatchSizeV_( 0 ), predPatchIndex_( 0 ) {}
PCCBitstreamDecoder::~PCCBitstreamDecoder() {}

int32_t PCCBitstreamDecoder::decode( PCCBitstream& bitstream, PCCContext& context ) {
  bitstream.getBitStreamStat().newGOF();
  VPCCUnitType vpccUnitType;
  vpccUnit( context, bitstream, vpccUnitType );  // VPCC_VPS
  vpccUnit( context, bitstream, vpccUnitType );  // VPCC_AD
  vpccUnit( context, bitstream, vpccUnitType );  // VPCC_OVD
  vpccUnit( context, bitstream, vpccUnitType );  // VPCC_GVD
  vpccUnit( context, bitstream, vpccUnitType );  // VPCC_AVD
  std::cout << " occupancy map  ->" << bitstream.getBitStreamStat().getTotalMetadata() << " B " << std::endl;
  return 1;
}

void PCCBitstreamDecoder::videoSubStream( PCCContext&   context,
                                             PCCBitstream& bitstream,
                                             VPCCUnitType& vpccUnitType ) {
  TRACE_BITSTREAM( "%s \n", __func__ );
  size_t atlasIndex = 0;
  if ( vpccUnitType == VPCC_OVD ) {
    TRACE_BITSTREAM( "OccupancyMap \n" );
    bitstream.read( context.createVideoBitstream( VIDEO_OCCUPANCY ) );
  } else if ( vpccUnitType == VPCC_GVD ) {
    TRACE_BITSTREAM( "Geometry \n" );
    auto& sps = context.getSps();
    if ( sps.getMapCountMinus1(atlasIndex) > 0 && !sps.getMapAbsoluteCodingEnableFlag( atlasIndex, 1 ) ) {
      bitstream.read( context.createVideoBitstream( VIDEO_GEOMETRY_D0 ) );
      bitstream.read( context.createVideoBitstream( VIDEO_GEOMETRY_D1 ) );
    } else {
      bitstream.read( context.createVideoBitstream( VIDEO_GEOMETRY ) );
    }
    if ( sps.getRawPatchEnabledFlag(atlasIndex) && sps.getRawSeparateVideoPresentFlag(atlasIndex) ) {
      bitstream.read( context.createVideoBitstream( VIDEO_GEOMETRY_MP ) );
    }
  } else if ( vpccUnitType == VPCC_AVD ) {
    auto& sps = context.getSps();
    if ( sps.getAttributeInformation(atlasIndex).getAttributeCount() > 0 ) {
      TRACE_BITSTREAM( "Texture \n" );
      bitstream.read( context.createVideoBitstream( VIDEO_TEXTURE ) );
      if ( sps.getRawPatchEnabledFlag(atlasIndex) && sps.getRawSeparateVideoPresentFlag(atlasIndex) ) {
        bitstream.read( context.createVideoBitstream( VIDEO_TEXTURE_MP ) );
      }
    }
  }
}

// 7.3.2.1 General V-PCC unit syntax
void PCCBitstreamDecoder::vpccUnit( PCCContext& context, PCCBitstream& bitstream, VPCCUnitType& vpccUnitType ) {
  TRACE_BITSTREAM( "%s \n", __func__ );
  int32_t position = bitstream.size();
  vpccUnitHeader( context, bitstream, vpccUnitType );
  vpccUnitPayload( context, bitstream, vpccUnitType );
  // while( more_data_in_vpcc_unit() ) { bitstream.write( 0, 8 ); }
  bitstream.getBitStreamStat().setVpccUnitSize( vpccUnitType, (int32_t)bitstream.size() - position );
}

// 7.3.2.2 V-PCC unit header syntax
void PCCBitstreamDecoder::vpccUnitHeader( PCCContext& context, PCCBitstream& bitstream, VPCCUnitType& vpccUnitType ) {
  TRACE_BITSTREAM( "%s \n", __func__ );
  auto& vpcc   = context.getVPCC();
  vpccUnitType = (VPCCUnitType)bitstream.read( 5 );  // u(5)
  if ( vpccUnitType == VPCC_AVD || vpccUnitType == VPCC_GVD || vpccUnitType == VPCC_OVD || vpccUnitType == VPCC_AD ) {
    vpcc.setVpccParameterSetId( bitstream.read( 4 ) );  // u(4)
    vpcc.setAtlasId( bitstream.read( 6 ) );                 // u(6)
  }
  if ( vpccUnitType == VPCC_AVD ) {
    auto& sps = context.getSps();
    vpcc.setAttributeIndex( bitstream.read( 7 ) );           // u(7)
    vpcc.setAttributeDimensionIndex( bitstream.read( 5 ) );  // u(5)
    vpcc.setMapIndex( bitstream.read( 4 ) );                 // u(4)
    vpcc.setRawVideoFlag( bitstream.read( 1 ) );             // u(1)
  } else if ( vpccUnitType == VPCC_GVD ) {
    auto& sps = context.getSps();
    vpcc.setMapIndex( bitstream.read( 4 ) );      // u(4)
    vpcc.setRawVideoFlag( bitstream.read( 1 ) );  // u(1)
    bitstream.read( 12 );                         // u(12)
  } else if ( vpccUnitType == VPCC_OVD || vpccUnitType == VPCC_AD ) {
    bitstream.read( 17 );  // u(17)
  } else {
    bitstream.read( 27 );  // u(27)
  }
}

// 7.3.2.3 V-PCC unit payload syntax
void PCCBitstreamDecoder::vpccUnitPayload( PCCContext& context, PCCBitstream& bitstream, VPCCUnitType& vpccUnitType ) {
  TRACE_BITSTREAM( "%s \n", __func__ );
  TRACE_BITSTREAM( "vpccUnitType = %d \n", (int32_t)vpccUnitType );
  if ( vpccUnitType == VPCC_VPS ) {
    auto& sps = context.addVpccParameterSet( context.getVPCC().getVpccParameterSetId() );
    vpccParameterSet( sps, context, bitstream );
  } else if ( vpccUnitType == VPCC_AD ) {
    atlasSubStream( context, bitstream );
  } else if ( vpccUnitType == VPCC_OVD || vpccUnitType == VPCC_GVD || vpccUnitType == VPCC_AVD ) {
    videoSubStream( context, bitstream, vpccUnitType );
  }
}

// 7.3.3 Byte alignment syntax
void PCCBitstreamDecoder::byteAlignment( PCCBitstream& bitstream ) {
  TRACE_BITSTREAM( "%s \n", __func__ );
  bitstream.read( 1 );  // f(1): equal to 1
  while ( !bitstream.byteAligned() ) {
    bitstream.read( 1 );  // f(1): equal to 0
  }
}

// 7.3.4.1 General V-PCC parameter set syntax
void PCCBitstreamDecoder::vpccParameterSet( VpccParameterSet& sps,
                                            PCCContext&           context,
                                            PCCBitstream&         bitstream ) {
  TRACE_BITSTREAM( "%s \n", __func__ );
  profileTierLevel( sps.getProfileTierLevel(), bitstream );
  sps.setVpccParameterSetId( bitstream.read( 4 ) );  // u(4)
  sps.setAtlasCountMinus1( bitstream.read( 6 ) );    // u(6)
  sps.allocateAltas();
  for ( int j = 0; j < sps.getAtlasCountMinus1() + 1; j++ ) {
    TRACE_BITSTREAM( "Atlas = %lu \n", j );
    sps.setFrameWidth( j, bitstream.read( 16 ) );     // u(16)
    sps.setFrameHeight( j, bitstream.read( 16 ) );    // u(16)
    sps.setMapCountMinus1( j, bitstream.read( 4 ) );  // u(4)
    TRACE_BITSTREAM( " MapCountMinus1 = %lu \n", sps.getMapCountMinus1( j ) );
    sps.allocateMap( j );
    if ( sps.getMapCountMinus1( j ) > 0 ) {       
      sps.setMultipleMapStreamsPresentFlag( j, bitstream.read( 1 ) );  // u(1)
      TRACE_BITSTREAM( "MultipleMapStreamsPresentFlag = %lu \n", sps.getMultipleMapStreamsPresentFlag( j ) );
    }
    sps.setMapAbsoluteCodingEnableFlag( j, 0, 1 );
    for ( size_t i = 1; i <= sps.getMapCountMinus1( j ); i++ ) {
      if ( sps.getMultipleMapStreamsPresentFlag( j ) ) {
        sps.setMapAbsoluteCodingEnableFlag( j, i, bitstream.read( 1 ) );  // u(1)
      } else {
        sps.setMapAbsoluteCodingEnableFlag( j, i, 1 );
      }
      if ( sps.getMapAbsoluteCodingEnableFlag( j, i ) == 0 ) {
        if ( i > 0 ) {
          sps.setMapPredictorIndexDiff( j, i, bitstream.readUvlc() );  // ue(v)
        } else {
          sps.setMapPredictorIndexDiff( j, i, 0 );
        }
      }
    }    
    sps.setRawPatchEnabledFlag( j, bitstream.read( 1 ) );  // u(1)
    TRACE_BITSTREAM( "RawPatchEnabledFlag = %lu \n", sps.getRawPatchEnabledFlag( j ) );
    if ( sps.getRawPatchEnabledFlag( j ) ) {
      sps.setRawSeparateVideoPresentFlag( j, bitstream.read( 1 ) );  // u(1)
      TRACE_BITSTREAM( "RawSeparateVideoPresentFlag = %lu \n", sps.getRawSeparateVideoPresentFlag( j ) );
    }
#ifdef BITSTREAM_TRACE
    for ( size_t i = 0; i < sps.getMapCountMinus1( j ) + 1; i++ ) {
      TRACE_BITSTREAM( "AbsoluteCoding L%lu = %lu \n", i, sps.getMapAbsoluteCodingEnableFlag( j, i ) );
    }
#endif
    occupancyInformation( sps.getOccupancyInformation( j ), bitstream );
    geometryInformation( sps.getGeometryInformation( j ), sps, bitstream );
    attributeInformation( sps.getAttributeInformation( j ), sps, bitstream );
  }

  sps.setExtensionPresentFlag( bitstream.read( 1 ) );  // u(1)
  if ( sps.getExtensionPresentFlag() ) {
    sps.setExtensionLength( bitstream.readUvlc() );  // ue(v)
    sps.allocateExtensionDataByte();
    for( size_t i=0;i<sps.getExtensionLength();i++){
      sps.setExtensionDataByte( i, bitstream.read( 8 ) ); // u(8)
    }
  }

  // THE NEXT PARAMETERS ARE NOT IN THE VPCC CD SYNTAX DOCUMENTS AND WILL BE REMOVE
#ifdef BITSTREAM_TRACE
  bitstream.trace( "  Depreciated1\n" );
#endif
  sps.setLosslessGeo444( bitstream.read( 1 ) );    // u(1) TODO: remove?
  sps.setLosslessGeo( bitstream.read( 1 ) );       // u(1) TODO: remove?

  sps.setMinLevel( bitstream.read( 8 ) );          // u(8) TODO: remove?
  sps.setSurfaceThickness( bitstream.read( 8 ) );  // u(8) TODO: remove?
  
  sps.setPatchInterPredictionEnabledFlag( bitstream.read( 1 ) );      // u(1) TODO: remove?
  sps.setPixelDeinterleavingFlag( bitstream.read( 1 ) );              // u(1) TODO: remove?
  sps.setPointLocalReconstructionEnabledFlag( bitstream.read( 1 ) );  // u(1) TODO: remove?
  if ( sps.getPointLocalReconstructionEnabledFlag() ) {// TODO: remove?
    pointLocalReconstructionInformation( sps.getPointLocalReconstructionInformation(), context, bitstream );// TODO: remove?
  }//  TODO: remove?
  sps.setRemoveDuplicatePointEnabledFlag( bitstream.read( 1 ) );  // u(1) TODO: remove?
  sps.setProjection45DegreeEnableFlag( bitstream.read( 1 ) );     // u(1) TODO: remove?
  sps.setPatchPrecedenceOrderFlag( bitstream.read( 1 ) );         // u(1) TODO: remove?
#ifdef BITSTREAM_TRACE
  bitstream.trace( "  Depreciated2\n" );
#endif
  // THE NEXT PARAMETERS ARE NOT IN THE VPCC CD SYNTAX DOCUMENTS AND WILL BE REMOVE
  byteAlignment( bitstream );
}

// 7.3.4.2 Profile, tier, and level syntax
void PCCBitstreamDecoder::profileTierLevel( ProfileTierLevel& ptl, PCCBitstream& bitstream ) {
  TRACE_BITSTREAM( "%s \n", __func__ );
  ptl.setTierFlag( bitstream.read( 1 ) );                  // u(1)
  ptl.setProfileCodecGroupIdc( bitstream.read( 7 ) );      // u(7)
  ptl.setProfilePccToolsetIdc( bitstream.read( 8 ) );      // u(8)
  ptl.setProfileReconctructionIdc( bitstream.read( 8 ) );  // u(8)
  bitstream.read( 32 );                                    // u(32)
  ptl.setLevelIdc( bitstream.read( 8 ) );                  // u(8)
}

// 7.3.4.3 Occupancy parameter set syntax
void PCCBitstreamDecoder::occupancyInformation( OccupancyInformation& oi, PCCBitstream& bitstream ) {
  TRACE_BITSTREAM( "%s \n", __func__ );
  oi.setOccupancyCodecId( bitstream.read( 8 ) );                       // u(8)
  oi.setLossyOccupancyMapCompressionThreshold( bitstream.read( 8 ) );  // u(8)
  oi.setOccupancyNominal2DBitdepthMinus1( bitstream.read( 5 ) );      // u(5)
  oi.setOccupancyMSBAlignFlag( bitstream.read( 1 ) );                  // u(1)
  TRACE_BITSTREAM( "  OccupancyLossyThreshold = %d  \n", oi.getLossyOccupancyMapCompressionThreshold() );
}

// 7.3.4.4 Geometry parameter set syntax
void PCCBitstreamDecoder::geometryInformation( GeometryInformation&  gi,
                                               VpccParameterSet& sps,
                                               PCCBitstream&         bitstream ) {
  TRACE_BITSTREAM( "%s \n", __func__ );
  size_t atlasIndex = 0;
  gi.setGeometryCodecId( bitstream.read( 8 ) );                      // u(8)
  gi.setGeometryNominal2dBitdepthMinus1( bitstream.read( 5 ) );      // u(5)
  gi.setGeometryMSBAlignFlag( bitstream.read( 1 ) );                 // u(1)
  gi.setGeometry3dCoordinatesBitdepthMinus1( bitstream.read( 5 ) );  // u(5)
  if ( sps.getRawSeparateVideoPresentFlag( atlasIndex ) ) {
    gi.setRawGeometryCodecId( bitstream.read( 8 ) );  // u(8)
  }

  // JR TODO:  Remove
  gi.setGeometryParamsEnabledFlag( bitstream.read( 1 ) );       // u(1) TODO: remove?
  gi.setGeometryPatchParamsEnabledFlag( bitstream.read( 1 ) );  // u(1) TODO: remove?
  TRACE_BITSTREAM( "GeometryParamsEnabledFlag = %d \n", gi.getGeometryParamsEnabledFlag() );
  TRACE_BITSTREAM( "GeometryPatchParamsEnabledFlag = %d \n", gi.getGeometryPatchParamsEnabledFlag() );
}

// 7.3.4.5 Attribute information
void PCCBitstreamDecoder::attributeInformation( AttributeInformation& ai,
                                                VpccParameterSet& sps,
                                                PCCBitstream&         bitstream ) {
  TRACE_BITSTREAM( "%s \n", __func__ );
  size_t atlasIndex = 0; 
  ai.setAttributeCount( bitstream.read( 7 ) );  // u(7)
  TRACE_BITSTREAM( "AttributeCount = %u  \n", ai.getAttributeCount() );
  ai.allocate();
  for ( size_t i = 0; i < ai.getAttributeCount(); i++ ) {
    ai.setAttributeTypeId( i, bitstream.read( 4 ) );   // u(4)
    ai.setAttributeCodecId( i, bitstream.read( 8 ) );  // u(8)
    if ( sps.getRawSeparateVideoPresentFlag(atlasIndex) ) {
      ai.setRawAttributeCodecId( i, bitstream.read( 8 ) );  // u(8) 
    }
    for ( int32_t j = 0; j < sps.getMapCountMinus1( atlasIndex ); j++ ) {
      if ( sps.getMapAbsoluteCodingEnableFlag( atlasIndex, j ) == 0 ) {
        ai.setAttributeMapAbsoluteCodingEnabledFlag( i, bitstream.read( 1 ) );  // u(1)
      }
    }
    ai.setAttributeDimensionMinus1( i, bitstream.read( 6 ) );  // u(6)
    if ( ai.getAttributeDimensionMinus1( i ) > 0 ) {
      ai.setAttributeDimensionPartitionsMinus1( i, bitstream.read( 6 ) );  // u(6)
      int32_t remainingDimensions = ai.getAttributeDimensionMinus1( i );
      int32_t k                   = ai.getAttributeDimensionPartitionsMinus1( i );
      for ( int32_t j = 0; j < k; j++ ) {
        if ( k - j == remainingDimensions ) {
          ai.setAttributePartitionChannelsMinus1( i, j, 0 );
        } else {
          ai.setAttributePartitionChannelsMinus1( i, j, bitstream.readUvlc() );  // ue(v)
        }
        remainingDimensions -= ai.getAttributePartitionChannelsMinus1( i, j ) + 1;
      }
      ai.setAttributePartitionChannelsMinus1( i, k, remainingDimensions );
    }
    ai.setAttributeNominal2dBitdepthMinus1( i, bitstream.read( 5 ) );  // u(5)
  }
  if ( ai.getAttributeCount() > 0 ) {
    ai.setAttributeMSBAlignFlag( bitstream.read( 1 ) );             // u(1) 
    // ai.setAttributeParamsEnabledFlag( bitstream.read( 1 ) );       // u(1) TODO: remove?
    // ai.setAttributePatchParamsEnabledFlag( bitstream.read( 1 ) );  // u(1) TODO: remove?
  }
}

// 7.3.6.1 Atlas sequence parameter set RBSP
void PCCBitstreamDecoder::atlasSequenceParameterSetRBSP( AtlasSequenceParameterSetRBSP& asps,
                                                         PCCContext&                    context,
                                                         PCCBitstream&                  bitstream ) {
  asps.setAltasSequenceParameterSetId( bitstream.readUvlc() );  // ue(v)
  asps.setFrameWidth( bitstream.read( 16 ) );  // u(16)
  asps.setFrameHeight( bitstream.read( 16 ) );  // u(16)
  asps.setLog2PatchPackingBlockSize( bitstream.read( 3 ) );  // u(3)
  asps.setLog2MaxAtlasFrameOrderCntLsbMinus4( bitstream.readUvlc() );  // ue(v)
  asps.setMaxDecAtlasFrameBufferingMinus1( bitstream.readUvlc() );  // ue(v)
  asps.setLongTermRefAtlasFramesFlag( bitstream.read( 1 ) );  // u(1)
  asps.setNumRefAtlasFrameListsInAsps( bitstream.readUvlc() );  // ue(v)  
  asps.allocateRefListStruct();
  for ( size_t i = 0; i < asps.getNumRefAtlasFrameListsInAsps(); i++ ) {
    refListStruct( asps.getRefListStruct( i ), asps, bitstream );
  }
  asps.setLongTermRefAtlasFramesFlag( bitstream.read( 1 ) );  // u(1)
  asps.set45DegreeProjectionPatchPresentFlag( bitstream.read( 1 ) );  // u(1)
  asps.setNormalAxisLimitsQuantizationEnabledFlag( bitstream.read( 1 ) );  // u(1)
  asps.setNormalAxisMaxDeltaValueEnabledFlag( bitstream.read( 1 ) );  // u(1)
  asps.setRemoveDuplicatePointEnabledFlag( bitstream.read( 1 ) );  // u(1)
  asps.setPixelDeinterleavingFlag( bitstream.read( 1 ) );  // u(1)
  asps.setPatchPrecedenceOrderFlag( bitstream.read( 1 ) );  // u(1)
  asps.setPatchSizeQuantizerPresentFlag( bitstream.read( 1 ) );  // u(1)
  asps.setEnhancedOccupancyMapForDepthFlag( bitstream.read( 1 ) );  // u(1)
  asps.setPointLocalReconstructionEnabledFlag( bitstream.read( 1 ) );  // u(1)

  asps.setMapCountMinus1( bitstream.read( 4 ) );  // u(4)
  if( asps.getEnhancedOccupancyMapForDepthFlag() && asps.getMapCountMinus1() == 0 ){  
    asps.setEnhancedOccupancyMapFixBitCountMinus1( bitstream.read( 4 ) );  // u(4)
  }
  if( asps.getPointLocalReconstructionEnabledFlag() ){
    asps.allocatePointLocalReconstructionInformation();
    pointLocalReconstructionInformation( asps, context, bitstream ); 
  }
  if( asps.getPixelDeinterleavingFlag() || asps.getPointLocalReconstructionEnabledFlag() ){
    asps.setSurfaceThicknessMinus1( bitstream.read( 8 ) );  // u(8)
  }
  asps.setVuiParametersPresentFlag( bitstream.read( 1 ) );  // u(1)
  if( asps.getVuiParametersPresentFlag() ){
    vuiParameters(); // TODO
  }
  asps.setExtensionPresentFlag( bitstream.read( 1 ) );  // u(1)
  if( asps.getExtensionPresentFlag() ){
    while( moreRbspData(bitstream) ){
      asps.setExtensionPresentFlag( bitstream.read( 1 ) );  // u(1)    
    }
  }
  rbspTrailingBits(bitstream);
}

// 7.3.4.6 Point local reconstruction information syntax (OLD)
void PCCBitstreamDecoder::pointLocalReconstructionInformation( PointLocalReconstructionInformation& plri,
                                                               PCCContext&                          context,
                                                               PCCBitstream&                        bitstream ) {
  TRACE_BITSTREAM( "%s \n", __func__ );
  plri.setNumberOfModesMinus1( bitstream.read( 4 ) );  // u(4)
  TRACE_BITSTREAM( "  NumberOfModesMinus1 = %u \n", plri.getNumberOfModesMinus1() );
  plri.allocate();
  for ( size_t i = 0; i < plri.getNumberOfModesMinus1(); i++ ) {
    plri.setInterpolateFlag( i, bitstream.read( 1 ) );  // u(1)
    plri.setFillingFlag( i, bitstream.read( 1 ) );      // u(1)
    plri.setMinimumDepth( i, bitstream.read( 2 ) );     // u(2)
    plri.setNeighbourMinus1( i, bitstream.read( 2 ) );  // u(2)
    TRACE_BITSTREAM( "  Mode[%lu] = I = %d F = %d D = %u N = %u \n", i, plri.getInterpolateFlag( i ),
                     plri.getFillingFlag( i ), plri.getMinimumDepth( i ), plri.getNeighbourMinus1( i ) );
  }
  plri.setBlockThresholdPerPatchMinus1( bitstream.readUvlc() );  // ue(v)
  TRACE_BITSTREAM( "  BlockThresholdPerPatchMinus1 = %u \n", plri.getBlockThresholdPerPatchMinus1() );
}

// 7.3.6.2 Point local reconstruction information syntax (NEW)
void PCCBitstreamDecoder::pointLocalReconstructionInformation( AtlasSequenceParameterSetRBSP& asps,
                                                               PCCContext&                          context,
                                                               PCCBitstream&                        bitstream ) {
  TRACE_BITSTREAM( "%s \n", __func__ );
  asps.allocatePointLocalReconstructionInformation();
  for( size_t j = 0; j < asps.getMapCountMinus1() + 1 ; j ++ ) {
    auto& plri = asps.getPointLocalReconstructionInformation( j );
    plri.setMapEnabledFlag( bitstream.read( 1 ) );  // u(1)
    if( plri.getMapEnabledFlag() ) {
      plri.setNumberOfModesMinus1( bitstream.read( 4 ) );  // u(4)
      TRACE_BITSTREAM( "  NumberOfModesMinus1 = %u \n", plri.getNumberOfModesMinus1() );
      plri.allocate();
      for ( size_t i = 0; i < plri.getNumberOfModesMinus1(); i++ ) {
        plri.setInterpolateFlag( i, bitstream.read( 1 ) );  // u(1)
        plri.setFillingFlag( i, bitstream.read( 1 ) );      // u(1)
        plri.setMinimumDepth( i, bitstream.read( 2 ) );     // u(2)
        plri.setNeighbourMinus1( i, bitstream.read( 2 ) );  // u(2)
        TRACE_BITSTREAM( "  Mode[%lu] = I = %d F = %d D = %u N = %u \n", i, plri.getInterpolateFlag( i ),
                        plri.getFillingFlag( i ), plri.getMinimumDepth( i ), plri.getNeighbourMinus1( i ) );
      }
      plri.setBlockThresholdPerPatchMinus1( bitstream.read( 6 ) );  // u(6)
      TRACE_BITSTREAM( "  BlockThresholdPerPatchMinus1 = %u \n", plri.getBlockThresholdPerPatchMinus1() );
    }
  }
}
//jkei: newly added PST:Oct30th
bool PCCBitstreamDecoder::moreRbspData( PCCBitstream& bitstream ){
  //jkei: TODO: need to fill this part
  return false;
}
void PCCBitstreamDecoder::rbspTrailingBits( PCCBitstream& bitstream ) {
  //jkei: TODO: need to fill this part
  TRACE_BITSTREAM( "%s \n", __func__ );
  bitstream.read( 1 );  // f(1): equal to 1
  while ( !bitstream.byteAligned() ) {
    bitstream.read( 1 );  // f(1): equal to 0
  }
}

//7.3.6.3  Atlas frame parameter set RBSP syntax
void PCCBitstreamDecoder::atlasFrameParameterSetRbsp(AtlasFrameParameterSetRbsp& afps, PCCContext& context, PCCBitstream& bitstream ){
  TRACE_BITSTREAM( "%s \n", __func__ );
  afps.setAtlasFrameParameterSetId( bitstream.readUvlc() );  //ue(v)
  afps.setAtlasSequenceParameterSetId(bitstream.readUvlc());//ue(v)
  atlasFrameTileInformation(afps.getAtlasFrameTileInformation(), context.getSps(), bitstream );
  
  afps.setAfpsNumRefIdxDefaultActiveMinus1( bitstream.readUvlc()  ); //ue(v)
  afps.setAfpsAdditionalLtAfocLsbLen(bitstream.readUvlc() ); //ue(v)
  afps.setAfps2dPosXBitCountMinus1( bitstream.read(4) );
  afps.setAfps2dPosYBitCountMinus1(  bitstream.read(4) );
  afps.setAfps3dPosXBitCountMinus1(  bitstream.read(5) );
  afps.setAfps3dPosYBitCountMinus1(  bitstream.read(5) );
  afps.setAfpsOverrideEomForDepthFlag(  bitstream.read(1) );
  if(afps.getAfpsOverrideEomForDepthFlag()){
    afps.setAfpsEomNumberOfPatchBitCountMinus1( bitstream.read(4) );
    afps.setAfpsEomMaxBitCountMinus1(bitstream.read(4) );
  }
  afps.setAfpsRaw3dPosBitCountExplicitModeFlag( bitstream.read(1) );
  afps.setAfpsExtensionPresentFlag( bitstream.read(1));
  if(afps.getAfpsExtensionPresentFlag()){
    while( moreRbspData(bitstream) )
    {
      afps.setAfpsExtensionDataFlag( bitstream.read(1) );
    }
  }
  rbspTrailingBits(bitstream);
}

//7.3.6.4  Atlas frame tile information syntax
void PCCBitstreamDecoder::atlasFrameTileInformation( AtlasFrameTileInformation& afti, VpccParameterSet& sps, PCCBitstream& bitstream )
{
  TRACE_BITSTREAM( "%s \n", __func__ );
  afti.setSingleTileInAtlasFrameFlag( bitstream.read( 1 ) );  // u(1)
  if ( !afti.getSingleTileInAtlasFrameFlag() ) {
    afti.setUniformTileSpacingFlag( bitstream.read( 1 ) );  // u(1)
    if ( afti.getUniformTileSpacingFlag() ) {
      afti.setTileColumnWidthMinus1( 0, bitstream.readUvlc() );  //  ue(v)
      afti.setTileRowHeightMinus1( 0, bitstream.readUvlc() );    //  ue(v)
    } else {
      afti.setNumTileColumnsMinus1( bitstream.readUvlc() );  //  ue(v)
      afti.setNumTileRowsMinus1( bitstream.readUvlc() );     //  ue(v)
      for ( size_t i = 0; i < afti.getNumTileColumnsMinus1(); i++ ) {
        afti.setTileColumnWidthMinus1( i, bitstream.readUvlc() );  //  ue(v)
      }
      for ( size_t i = 0; i < afti.getNumTileRowsMinus1(); i++ ) {
        afti.setTileRowHeightMinus1( i, bitstream.readUvlc() );  //  ue(v)
      }
    }
  }

  afti.setSingleTilePerTileGroupFlag( bitstream.read( 1 ) );  //  u(1)
  if ( !afti.getSingleTilePerTileGroupFlag() ) {
    uint32_t NumTilesInPatchFrame     = ( afti.getNumTileColumnsMinus1() + 1 ) * ( afti.getNumTileRowsMinus1() + 1 );

    afti.setNumTileGroupsInAtlasFrameMinus1( bitstream.readUvlc() );  // ue(v)
    for ( size_t i = 0; i <= afti.getNumTileGroupsInAtlasFrameMinus1(); i++ ) {
      uint8_t bitCount = getFixedLengthCodeBitsCount( NumTilesInPatchFrame + 1 );
      if ( i > 0 ) {
        afti.setTopLeftTileIdx( i, bitstream.read( bitCount ) );
      }  // u(v) : Ceil( Log2( NumTilesInPatchFrame )
      bitCount = getFixedLengthCodeBitsCount( NumTilesInPatchFrame - afti.getTopLeftTileIdx( i ) + 1 );
      afti.setBottomRightTileIdxDelta(
          i, bitstream.read( bitCount ) );  // u(v) : Ceil( Log2( NumTilesInPatchFrame − pfti_top_left_tile_idx[ i ] ) )
    }
  }
  afti.setSignalledTileGroupIdFlag( bitstream.read( 1 ) );  // u(1)
  if ( afti.getSignalledTileGroupIdFlag() ) {
    afti.setSignalledTileGroupIdLengthMinus1( bitstream.readUvlc() );  // ue(v)
    for ( size_t i = 0; i <= afti.getSignalledTileGroupIdLengthMinus1(); i++ ) {
      uint8_t bitCount = afti.getSignalledTileGroupIdLengthMinus1() + 1;
      afti.setTileGroupId( i,
                           bitstream.read( bitCount ) );  // u(v) : pfti_signalled_tile_group_id_length_minus1 + 1  bits
      // When not present, the value of pfti_tile_group_id[ i ] is inferred to be equal to i, for each i in the range of
      // 0 to pfti_num_tile_groups_in_patch_frame_minus1, inclusive.
    }
  }
}


//7.3.6.5  Supplemental enhancement information RBSP syntax
//7.3.6.6  Access unit delimiter RBSP syntax
//7.3.6.7  End of sequence RBSP syntax
//7.3.6.8  End of bitstream RBSP syntax
//7.3.6.9  Filler data RBSP syntax

//7.3.6.10  Atlas tile group layer RBSP syntax = patchTileGroupLayerUnit
void PCCBitstreamDecoder::atlasTileGroupLayerRbsp (AtlasTileGroupLayerRbsp& atgl, PCCContext& context, PCCBitstream& bitstream ){
  //setFrameIndex
  atlasTileGroupHeader(atgl.getAtlasTileGroupHeader(), context, bitstream);
  if( atgl.getAtlasTileGroupHeader().getAtghType() != SKIP_TILE_GRP )
    atlasTileGroupDataUnit(atgl.getAtlasTileGroupDataUnit(), atgl.getAtlasTileGroupHeader(), context, bitstream);
  rbspTrailingBits(bitstream);
}

//7.3.6.11  Atlas tile group header syntax
void PCCBitstreamDecoder::atlasTileGroupHeader(AtlasTileGroupHeader& atgh, PCCContext& context, PCCBitstream& bitstream ){
  atgh.setAtghAtlasFrameParameterSetId( bitstream.readUvlc() );
  ////jkei
  AtlasFrameParameterSetRbsp afps; //<<from setAtghAtlasFrameParameterSetId!!
  AtlasSequenceParameterSetRBSP asps; //<<from afps.getSeqeucneParameterSetId()
  AtlasFrameTileInformation& afti = afps.getAtlasFrameTileInformation();
  ///

  atgh.setAtghAddress( bitstream.read(afti.getSignalledTileGroupIdLengthMinus1() + 1) );
  atgh.setAtghType( bitstream.readUvlc() );
  atgh.setAtghAtlasFrmOrderCntLsb( bitstream.read(asps.getLog2MaxAtlasFrameOrderCntLsbMinus4() + 4));
  if( asps.getNumRefAtlasFrameListsInAsps() > 0 )
    atgh.setAtghRefAtlasFrameListSpsFlag( bitstream.read(1));
  if( atgh.getAtghRefAtlasFrameListSpsFlag()==0 ){
    atgh.allocate(asps.getNumRefAtlasFrameListsInAsps());
    for ( size_t i = 0; i < asps.getNumRefAtlasFrameListsInAsps(); i++ ) {
      refListStruct( atgh.getRefListStruct( i ), asps, bitstream );
    }
  }
  else if( atgh.getAtghRefAtlasFrameListSpsFlag() > 0 ){
    size_t bitCount = getFixedLengthCodeBitsCount( asps.getNumRefAtlasFrameListsInAsps()+1); //by Ceil( Log2( asps_num_ref_atlas_frame_lists_in_asps ) )
    atgh.setAtghRefAtlasFrameListIdx( bitstream.read(bitCount ) );
  }
  
  uint8_t rlsIdx =
      asps.getNumRefAtlasFrameListsInAsps() ? atgh.getAtghRefAtlasFrameListIdx() :asps.getNumRefAtlasFrameListsInAsps();
  size_t numLtrAtlasFrmEntries = 0;
  for ( size_t i = 0; i < asps.getRefListStruct( rlsIdx ).getNumRefEntries(); i++ ) {
    if ( !atgh.getRefListStruct( rlsIdx ).getStRefAtalsFrameFlag(i) ) { numLtrAtlasFrmEntries++; }
  }
  
  for(size_t j = 0; j < numLtrAtlasFrmEntries; j++ ) {
    atgh.setAtghAdditionalAfocLsbPresentFlag(j, bitstream.read( 1 ));
    //afps_additional_lt_afoc_lsb_len
    if(atgh.getAtghAdditionalAfocLsbPresentFlag(j)){
      uint8_t bitCount = afps.getAfpsAdditionalLtAfocLsbLen();
      atgh.setAtghAdditionalAfocLsbVal(j, bitstream.read( bitCount ));
    }
  }
  if( atgh.getAtghType() != SKIP_TILE_GRP ) {
    if( asps.getNormalAxisLimitsQuantizationEnabledFlag() ) {
      atgh.setAtghPosMinZQuantizer( bitstream.read( 5 ) ) ;
      if( asps.getNormalAxisMaxDeltaValueEnabledFlag())
        atgh.setAtghPosDeltaMaxZQuantizer(bitstream.read( 5 ));
    }
    if( asps.getPatchSizeQuantizerPresentFlag() ) {
      atgh.setAtghPatchSizeXinfoQuantizer(bitstream.read(3));
      atgh.setAtghPatchSizeYinfoQuantizer(bitstream.read(3));
    }
    //0??
    size_t geometry3dBitdepthMinus1 =context.getSps().getGeometryInformation(0).getGeometry3dCoordinatesBitdepthMinus1();
     size_t geometry2dBitdepthMinus1 =context.getSps().getGeometryInformation(0).getGeometryNominal2dBitdepthMinus1();
    if( afps.getAfpsRaw3dPosBitCountExplicitModeFlag()){
      size_t bitCount = getFixedLengthCodeBitsCount(geometry3dBitdepthMinus1+1);
        atgh.setAtghRaw3dPosAxisBitCountMinus1( bitstream.read(bitCount));
    }
    else{
      atgh.setAtghRaw3dPosAxisBitCountMinus1( geometry3dBitdepthMinus1 - geometry2dBitdepthMinus1-1);
    }
    if( atgh.getAtghType() == P_TILE_GRP  && atgh.getRefListStruct(rlsIdx).getNumRefEntries() > 1 )
    {
      atgh.setAtghNumRefIdxActiveOverrideFlag( bitstream.read(1) );
      if( atgh.getAtghNumRefIdxActiveOverrideFlag())
        atgh.setAtghNumRefdxActiveMinus1( bitstream.readUvlc() );
    }
  }
  byteAlignment(bitstream);
}

//7.3.6.12  Reference list structure syntax (NEW)
void PCCBitstreamDecoder::refListStruct( RefListStruct&             rls,
                                         AtlasSequenceParameterSetRBSP& asps,
                                         PCCBitstream&              bitstream ) {
  TRACE_BITSTREAM( "%s \n", __func__ );
  rls.setNumRefEntries( bitstream.readUvlc() );  // ue(v)
  TRACE_BITSTREAM( "NumRefEntries = %lu  \n", rls.getNumRefEntries() );
  rls.allocate();
  for ( size_t i = 0; i < rls.getNumRefEntries(); i++ ) {
    if ( asps.getLongTermRefAtlasFramesFlag() ) {
      rls.setStRefAtalsFrameFlag( i, bitstream.read( 1 ) );  // u(1)
      if ( rls.getStRefAtalsFrameFlag( i ) ) {
        rls.setAbsDeltaPfocSt( i, bitstream.readUvlc() );  // ue(v)
        if ( rls.getAbsDeltaPfocSt( i ) > 0 ) {
          rls.setStrpfEntrySignFlag( i, bitstream.read( 1 ) );  // u(1)
        } else {
          uint8_t bitCount = asps.getLog2MaxAtlasFrameOrderCntLsbMinus4() + 4;
          rls.setAfocLsbLt(
              i, bitstream.read( bitCount ) );  // u(v) : psps_log2_max_patch_frame_order_cnt_lsb_minus4  + 4 bits
        }
      }
    }
  }
  TRACE_BITSTREAM( "%s done \n", __func__ );
}

//7.3.7.1  General atlas tile group data unit syntax =patchTileGroupDataUnit
void PCCBitstreamDecoder::atlasTileGroupDataUnit(AtlasTileGroupDataUnit& atgdu, AtlasTileGroupHeader& atgh, PCCContext& context, PCCBitstream& bitstream){
    TRACE_BITSTREAM( "%s \n", __func__ );
    TRACE_BITSTREAM( "atgh.getAtghType()        = %lu \n", atgh.getAtghType() );
    atgdu.init();
    size_t patchIndex = 0;
    auto tileGroupType = atgh.getAtghType();
    uint8_t patchMode  = bitstream.readUvlc();  // ue(v)
    TRACE_BITSTREAM( "patchMode = %lu \n", patchMode );
    while ( !( ( ( PCCTILEGROUP(tileGroupType) == I_TILE_GRP ) && ( patchMode == PATCH_MODE_I_END ) ) ||
               ( ( PCCTILEGROUP(tileGroupType) == P_TILE_GRP ) && ( patchMode == PATCH_MODE_P_END ) ) ) ) {
      auto& pid = atgdu.addPatchInformationData( patchMode );
      pid.setFrameIndex( atgdu.getFrameIndex() );
      pid.setPatchIndex( patchIndex );
      patchIndex++;
      patchInformationData( pid, patchMode, atgh, context, bitstream );
      patchMode = bitstream.readUvlc();  // ue(v)
      TRACE_BITSTREAM( "patchMode = %lu \n", patchMode );
    }
    TRACE_BITSTREAM( "ptgdu.getPatchCount() = %lu \n", atgdu.getPatchCount() );
    byteAlignment( bitstream );
  }

//7.3.7.2  Patch information data syntax
void PCCBitstreamDecoder::patchInformationData( PatchInformationData& pid, size_t patchMode, AtlasTileGroupHeader& atgh, PCCContext& context, PCCBitstream& bitstream ){
  TRACE_BITSTREAM( "%s \n", __func__ );
  auto& sps  = context.getSps();
  size_t atlasIndex = 0;
  auto& ai   = sps.getAttributeInformation(atlasIndex);
  pid.allocate( ai.getAttributeCount() );
  if ( ( PCCTILEGROUP( atgh.getAtghType() ) ) == P_TILE_GRP && patchMode == PATCH_MODE_P_SKIP ) {
  }else if( ( PCCTILEGROUP( atgh.getAtghType() ) ) == P_TILE_GRP && patchMode == PATCH_MODE_P_MERGE ) {
    auto& mpdu = pid.getMergePatchDataUnit();
    mpdu.setFrameIndex( pid.getFrameIndex() );
    mpdu.setPatchIndex( pid.getPatchIndex() );
    mergePatchDataUnit( mpdu, atgh, context, bitstream );
  }else if ( ( PCCTILEGROUP( atgh.getAtghType() ) ) == P_TILE_GRP && patchMode == PATCH_MODE_P_INTER ) {
    auto& ipdu = pid.getInterPatchDataUnit();
    ipdu.setFrameIndex( pid.getFrameIndex() );
    ipdu.setPatchIndex( pid.getPatchIndex() );
    interPatchDataUnit( ipdu, atgh, context, bitstream );
  } else if ( ( ( PCCTILEGROUP( atgh.getAtghType() ) ) == I_TILE_GRP && patchMode == PATCH_MODE_I_INTRA ) ||
             ( ( PCCTILEGROUP( atgh.getAtghType() ) ) == P_TILE_GRP && patchMode == PATCH_MODE_P_INTRA ) ) {
    auto& pdu = pid.getPatchDataUnit();
    pdu.setFrameIndex( pid.getFrameIndex() );
    pdu.setPatchIndex( pid.getPatchIndex() );
    patchDataUnit( pdu, atgh, context, bitstream );
  }  else if ( ( ( PCCTILEGROUP( atgh.getAtghType() ) ) == I_TILE_GRP && patchMode == PATCH_MODE_I_Raw ) ||
              ( ( PCCTILEGROUP( atgh.getAtghType() ) ) == P_TILE_GRP && patchMode == PATCH_MODE_P_Raw ) ) {
    auto& rpdu = pid.getRawPatchDataUnit();
    rpdu.setFrameIndex( pid.getFrameIndex() );
    rpdu.setPatchIndex( pid.getPatchIndex() );
    rawPatchDataUnit( rpdu, atgh, context, bitstream );
  } else if ( ( ( PCCTILEGROUP( atgh.getAtghType() ) ) == I_TILE_GRP && patchMode == PATCH_MODE_I_EOM ) ||
             ( ( PCCTILEGROUP( atgh.getAtghType() ) ) == P_TILE_GRP && patchMode == PATCH_MODE_P_EOM ) ) {
    auto& epdu = pid.getEomPatchDataUnit();
    epdu.setFrameIndex( pid.getFrameIndex() );
    epdu.setPatchIndex( pid.getPatchIndex() );
    eomPatchDataUnit( epdu, atgh, context, bitstream );
  }
}

//7.3.7.3  Patch data unit syntax : AtlasTileGroupHeader instead of PatchTileGroupHeader
void PCCBitstreamDecoder::patchDataUnit( PatchDataUnit& pdu, AtlasTileGroupHeader& atgh, PCCContext& context, PCCBitstream& bitstream ){
  //////////////////////////
  size_t atlasFrameParameterSetId = atgh.getAtghAtlasFrameParameterSetId();
  AtlasFrameParameterSetRbsp afps;
  size_t atlasSequenceParameterSetId = afps.getAtlasSequenceParameterSetId();
  AtlasSequenceParameterSetRBSP asps;
  /////////////////////////
  TRACE_BITSTREAM( "%s \n", __func__ );
  
  pdu.setPdu2dPosX( bitstream.read( afps.getAfps2dPosXBitCountMinus1() + 1 ) );  // u(v)
  pdu.setPdu2dPosY( bitstream.read( afps.getAfps2dPosYBitCountMinus1() + 1 ) );  // u(v)
  pdu.setPdu2dDeltaSizeX( bitstream.readSvlc() );                                                 // se(v)
  pdu.setPdu2dDeltaSizeY( bitstream.readSvlc() );                                                 // se(v)
  pdu.setPdu3dPosX(
      bitstream.read( afps.getAfps3dPosXBitCountMinus1() + 1 ) );  // u(v)
  pdu.setPdu3dPosY(
      bitstream.read( afps.getAfps3dPosYBitCountMinus1() + 1 ) );  // u(v)
   const uint8_t bitCountForMinDepth =
   context.getSps().getGeometryInformation(0).getGeometry3dCoordinatesBitdepthMinus1() +1;
  pdu.setPdu3dPosMinZ(bitstream.read( bitCountForMinDepth ) );  // u(v)
  if ( asps.getNormalAxisMaxDeltaValueEnabledFlag() ) {
    uint8_t bitCountForMaxDepth = bitCountForMinDepth;
    if ( asps.get45DegreeProjectionPatchPresentFlag())
      bitCountForMaxDepth++;
    pdu.setPdu3dPosDeltaMaxZ( bitstream.read( bitCountForMaxDepth ) );  // u(v)
  }
  pdu.setPduProjectionId( bitstream.read( 1 ) );
  pdu.setPduOrientationIndex( bitstream.read( 3 ) );  // u(3)
  if ( afps.getLodModeEnableFlag()) {
    pdu.setLodEnableFlag(bitstream.read( 1 ) ); //u1
    if( pdu.getLodEnableFlag() ) {
      pdu.setLodScaleXminus1( uint8_t( bitstream.readUvlc()) ); //uev
      pdu.setLodScaleY( uint8_t( bitstream.readUvlc()) ); //uev
    }
  }

  if ( asps.getPointLocalReconstructionEnabledFlag() ) {
    pointLocalReconstructionData( pdu.getPointLocalReconstructionData(), context, bitstream );
  }
  TRACE_BITSTREAM( "Patch(%zu/%zu) => UV %4lu %4lu S=%4ld %4ld P=%zu O=%d A=%lu %lu %lu P45= %d %d lod=(%lu) %lu %lu\n ",
                  pdu.getPdu2dPosX(),
                  pdu.getPdu2dPosY(),
                  pdu.getPdu2dDeltaSizeX(),
                  pdu.getPdu2dDeltaSizeY(),
                  pdu.getPdu3dPosX(),
                  pdu.getPdu3dPosY(),
                  pdu.getPdu3dPosMinZ(),
                  pdu.getPdu3dPosDeltaMaxZ(),
                  pdu.getPduProjectionId(),
                  pdu.getPduOrientationIndex(),
                  pdu.getLodEnableFlag(),
                  pdu.getLodScaleXminus1(),
                  pdu.getLodScaleY());
}

//7.3.7.4  Skip patch data unit syntax
void PCCBitstreamDecoder::skipPatchDataUnit( SkipPatchDataUnit& spdu, AtlasTileGroupHeader& atgh, PCCContext& context, PCCBitstream& bitstream ){
  
}

//7.3.7.5  Merge patch data unit syntax
void PCCBitstreamDecoder::mergePatchDataUnit(MergePatchDataUnit& mpdu, AtlasTileGroupHeader& atgh, PCCContext& context, PCCBitstream& bitstream){
    //////////////////////////
    size_t atlasFrameParameterSetId = atgh.getAtghAtlasFrameParameterSetId();
    AtlasFrameParameterSetRbsp afps;
    size_t atlasSequenceParameterSetId = afps.getAtlasSequenceParameterSetId();
    AtlasSequenceParameterSetRBSP asps;
    bool overridePlrFlag=false;
    /////////////////////////
    
    TRACE_BITSTREAM( "%s \n", __func__ );
    if(atgh.getAtghNumRefdxActiveMinus1()>0) mpdu.setMpduRefIndex( bitstream.readUvlc());
    mpdu.setMpduOverride2dParamsFlag( bitstream.read( 1 ));
    if(mpdu.getMpduOverride2dParamsFlag()){
      mpdu.setMpdu2dPosX( bitstream.readSvlc() );              // se(v)
      mpdu.setMpdu2dPosY( bitstream.readSvlc() );              // se(v)
      mpdu.setMpdu2dDeltaSizeX( bitstream.readSvlc() );              // se(v)
      mpdu.setMpdu2dDeltaSizeY( bitstream.readSvlc() );               // se(v)
      if(asps.getPointLocalReconstructionEnabledFlag()) overridePlrFlag=true;
    }else{
      mpdu.setMpduOverride3dParamsFlag( bitstream.read( 1 ));
      if(mpdu.getMpduOverride3dParamsFlag()){
        mpdu.setMpdu3dPosX( bitstream.readSvlc() );    // se(v)
        mpdu.setMpdu3dPosY( bitstream.readSvlc() );  // se(v)
        mpdu.setMpdu3dPosMinZ( bitstream.readSvlc() );  // se(v)
        if ( asps.getNormalAxisMaxDeltaValueEnabledFlag() ) {
          mpdu.setMpdu3dPosDeltaMaxZ( bitstream.readSvlc() );  // se(v)
        }
        if(asps.getPointLocalReconstructionEnabledFlag()) {
          overridePlrFlag = bitstream.read(1);
          mpdu.setMpduOverridePlrFlag( overridePlrFlag );
        }
      }
    }//else

  if ( overridePlrFlag && asps.getPointLocalReconstructionEnabledFlag() ) {
    pointLocalReconstructionData( mpdu.getPointLocalReconstructionData(), context, bitstream );
  }
    
    TRACE_BITSTREAM(
                    "%zu frame %zu MergePatch => Ref Frame Idx = %d ShiftUV = %ld %ld DeltaSize = %ld %ld Axis = %ld %ld %ld\n",
                    mpdu.getMpduRefIndex(),
                    mpdu.getMpdu2dPosX(),
                    mpdu.getMpdu2dPosY(),
                    mpdu.getMpdu2dDeltaSizeX(),
                    mpdu.getMpdu2dDeltaSizeY(),
                    mpdu.getMpdu3dPosX(),
                    mpdu.getMpdu3dPosY(),
                    mpdu.getMpdu3dPosMinZ(),
                    mpdu.getMpdu3dPosDeltaMaxZ());
  }

//7.3.7.6  Inter patch data unit syntax
void PCCBitstreamDecoder::interPatchDataUnit( InterPatchDataUnit& ipdu, AtlasTileGroupHeader& atgh, PCCContext& context, PCCBitstream& bitstream){
  auto& sps = context.getSps();
  //////////////////////////
  size_t atlasFrameParameterSetId = atgh.getAtghAtlasFrameParameterSetId();
  AtlasFrameParameterSetRbsp afps;
  size_t atlasSequenceParameterSetId = afps.getAtlasSequenceParameterSetId();
  AtlasSequenceParameterSetRBSP asps;
  /////////////////////////
  TRACE_BITSTREAM( "%s \n", __func__ );
  if(atgh.getAtghNumRefdxActiveMinus1()>0)
    ipdu.setIpduRefIndex( bitstream.readUvlc());
  ipdu.setIpduRefPatchIndex( bitstream.readSvlc() );
  ipdu.setIpdu2dPosX( bitstream.readSvlc() );              // se(v)
  ipdu.setIpdu2dPosY( bitstream.readSvlc() );              // se(v)
  ipdu.setIpdu2dDeltaSizeX( bitstream.readSvlc() );              // se(v)
  ipdu.setIpdu2dDeltaSizeY( bitstream.readSvlc() );               // se(v)
  ipdu.setIpdu3dPosX( bitstream.readSvlc() );    // se(v)
  ipdu.setIpdu3dPosY( bitstream.readSvlc() );  // se(v)
  ipdu.setIpdu3dPosMinZ( bitstream.readSvlc() );  // se(v)
  if ( asps.getNormalAxisMaxDeltaValueEnabledFlag() ) {
    ipdu.setIpdu3dPosDeltaMaxZ( bitstream.readSvlc() );  // se(v)
  }
  if ( asps.getPointLocalReconstructionEnabledFlag() ) {
    auto&  plrd      = ipdu.getPointLocalReconstructionData();
    auto&  pdg       = context.getPatchDataGroup();
    auto&  pfluPrev  = pdg.getPatchTileGroupLayerUnit( predFramePatchTileGroupLayerUnitIndex_ );
    auto&  pfhPrev   = pfluPrev.getPatchTileGroupHeader();
    auto&  pfduPrev  = pfluPrev.getPatchTileGroupDataUnit();
    auto&  pidPrev   = pfduPrev.getPatchInformationData( ipdu.getIpduRefPatchIndex() + predPatchIndex_ );
    auto   patchMode = pfduPrev.getPatchMode( ipdu.getIpduRefPatchIndex() + predPatchIndex_ );
    size_t sizeU     = ipdu.getIpdu2dDeltaSizeX();
    size_t sizeV     = ipdu.getIpdu2dDeltaSizeY();
    if ( ( ( PCCPatchFrameType( pfhPrev.getType() ) ) == PATCH_FRAME_I && patchMode == PATCH_MODE_I_INTRA ) ||
         ( ( PCCPatchFrameType( pfhPrev.getType() ) ) == PATCH_FRAME_P && patchMode == PATCH_MODE_P_INTRA ) ) {
      sizeU += pidPrev.getPatchDataUnit().getPointLocalReconstructionData().getBlockToPatchMapWidth();
      sizeV += pidPrev.getPatchDataUnit().getPointLocalReconstructionData().getBlockToPatchMapHeight();
    } else if ( ( PCCPatchFrameType( pfhPrev.getType() ) ) == PATCH_FRAME_P && patchMode == PATCH_MODE_P_INTER ) {
      sizeU += pidPrev.getDeltaPatchDataUnit().getPointLocalReconstructionData().getBlockToPatchMapWidth();
      sizeV += pidPrev.getDeltaPatchDataUnit().getPointLocalReconstructionData().getBlockToPatchMapHeight();
    }
    plrd.allocate( sizeU, sizeV );
    pointLocalReconstructionData( plrd, context, bitstream );
    prevPatchSizeU_ = sizeU;
    prevPatchSizeV_ = sizeV;
    predPatchIndex_ += ( ipdu.getIpduRefPatchIndex() + 1 );
  }

  TRACE_BITSTREAM(
      "%zu frame %zu DeltaPatch => DeltaIdx = %d ShiftUV = %ld %ld DeltaSize = %ld %ld Axis = %ld %ld %ld\n",
                  ipdu.getIpduRefIndex(),
                  ipdu.getIpduRefPatchIndex(),
                  ipdu.getIpdu2dPosX(),
                  ipdu.getIpdu2dPosY(),
                  ipdu.getIpdu2dDeltaSizeX(),
                  ipdu.getIpdu2dDeltaSizeY(),
                  ipdu.getIpdu3dPosX(),
                  ipdu.getIpdu3dPosY(),
                  ipdu.getIpdu3dPosMinZ(),
                  ipdu.getIpdu3dPosDeltaMaxZ());
}

//7.3.7.7  Raw patch data unit syntax
void PCCBitstreamDecoder::rawPatchDataUnit( RawPatchDataUnit& ppdu, AtlasTileGroupHeader& atgh, PCCContext& context, PCCBitstream& bitstream ) {
  auto& sps = context.getSps();
  TRACE_BITSTREAM( "%s \n", __func__ );
  //////////////////////////
  size_t atlasFrameParameterSetId = atgh.getAtghAtlasFrameParameterSetId();
  AtlasFrameParameterSetRbsp afps;
  size_t atlasSequenceParameterSetId = afps.getAtlasSequenceParameterSetId();
  AtlasSequenceParameterSetRBSP asps;
  /////////////////////////
  size_t atlasIndex = 0;
  if ( sps.getRawSeparateVideoPresentFlag(atlasIndex) ) { ppdu.setRpduPatchInRawVideoFlag( bitstream.read( 1 ) ); }  // u(1)
  
  ppdu.setRpdu2dPosX( bitstream.read( afps.getAfps2dPosXBitCountMinus1() + 1 ) );         // u(v)
  ppdu.setRpdu2dPosY( bitstream.read( afps.getAfps2dPosYBitCountMinus1() + 1 ) );         // u(v)
  ppdu.setRpdu2dDeltaSizeX( bitstream.readSvlc() );                                                        // se(v)
  ppdu.setRpdu2dDeltaSizeY( bitstream.readSvlc() );                                                        // se(v)
  ppdu.setRpdu3dPosX( bitstream.read( atgh.getAtghRaw3dPosAxisBitCountMinus1()+1 ) );          // u(v)
  ppdu.setRpdu3dPosY( bitstream.read( atgh.getAtghRaw3dPosAxisBitCountMinus1()+1 ) );        // u(v)
  ppdu.setRpdu3dPosZ( bitstream.read( atgh.getAtghRaw3dPosAxisBitCountMinus1()+1 ) );           // u(v)
  ppdu.setRpduRawPoints( bitstream.readUvlc() );
  TRACE_BITSTREAM(
      "Raw Patch => UV %4lu %4lu  S=%4ld %4ld  UVD1=%4ld %4ld %4ld NumPcmPoints=%lu PatchInRawVideoFlag=%d \n",
      ppdu.getRpdu2dPosX(), ppdu.getRpdu2dPosY(), ppdu.getRpdu2dDeltaSizeX(), ppdu.getRpdu2dDeltaSizeY(),
      ppdu.getRpdu3dPosX(), ppdu.getRpdu3dPosY(), ppdu.getRpdu3dPosZ(), ppdu.getRpduRawPoints(),
      ppdu.getRpduPatchInRawVideoFlag() );
}

// 7.3.6.x EOM patch data unit syntax
void PCCBitstreamDecoder::eomPatchDataUnit( EOMPatchDataUnit& epdu, AtlasTileGroupHeader& atgh, PCCContext& context, PCCBitstream& bitstream ) {
  TRACE_BITSTREAM( "%s \n", __func__ );
  //////////////////////////
  size_t atlasFrameParameterSetId = atgh.getAtghAtlasFrameParameterSetId();
  AtlasFrameParameterSetRbsp afps;
  size_t atlasSequenceParameterSetId = afps.getAtlasSequenceParameterSetId();
  AtlasSequenceParameterSetRBSP asps;
  /////////////////////////
  epdu.setEpdu2dPosX( bitstream.read( afps.getAfps2dPosXBitCountMinus1() + 1 ) );  // u(v)
  epdu.setEpdu2dPosY( bitstream.read( afps.getAfps2dPosXBitCountMinus1() + 1 ) );  // u(v)
  epdu.setEpdu2dDeltaSizeX( bitstream.readSvlc() );                                                 // se(v)
  epdu.setEpdu2dDeltaSizeY( bitstream.readSvlc() );                                                 // se(v)
  epdu.setEpduAssociatedPatchesCountMinus1( bitstream.read( 8) );
  for(size_t cnt=0; cnt<epdu.getEpduAssociatedPatchesCountMinus1()+1; cnt++)
  {
    epdu.setEpduAssociatedPatches( bitstream.read( 8 ), cnt);
    epdu.setEpduEomPointsPerPatch( bitstream.readUvlc(), cnt);  //???
  }
  TRACE_BITSTREAM( "EOM Patch => UV %4lu %4lu  N=%4ld\n", epdu.getEpdu2dPosX(),
                  epdu.getEpdu2dPosY(), epdu.getEpduAssociatedPatchesCountMinus1()+1);
  for(size_t cnt=0; cnt<epdu.getEpduAssociatedPatchesCountMinus1()+1; cnt++)
    TRACE_BITSTREAM( "%4ld, %4ld\n", epdu.getEpduAssociatedPatches()[cnt], epdu.getEpduEomPointsPerPatch()[cnt]);
}

//jkei -----> OLD

// 7.3.5.1 General patch data group unit syntax
void PCCBitstreamDecoder::atlasSubStream( PCCContext& context, PCCBitstream& bitstream ) {
  TRACE_BITSTREAM( "%s \n", __func__ );
  size_t i                               = 0;
  auto&  pdg                             = context.getPatchDataGroup();
  size_t frameCount                      = 0;
  size_t prevFrameindex                  = 0;
  predFramePatchTileGroupLayerUnitIndex_ = -1;
  do {
    PDGUnitType unitType = (PDGUnitType)bitstream.readUvlc();  // ue(v)
    TRACE_BITSTREAM( "%s (%u): frame = %u \n", strUnitType( unitType ).c_str(), (uint32_t)unitType, frameCount );
    atlasSubStreamUnitPayload( pdg, unitType, frameCount, context, bitstream );
    if ( unitType == PDG_PTGLU ) {
      frameCount++;
      prevFrameindex = i;
      predFramePatchTileGroupLayerUnitIndex_++;
    }
    i++;
  } while ( !bitstream.read( 1 ) );
  byteAlignment( bitstream );
}

// 7.3.5.2 Patch data group unit payload syntax
void PCCBitstreamDecoder::atlasSubStreamUnitPayload( PatchDataGroup& pdg,
                                                     PDGUnitType     unitType,
                                                     size_t          frameIndex,
                                                     PCCContext&     context,
                                                     PCCBitstream&   bitstream ) {
  TRACE_BITSTREAM( "%s \n", __func__ );
  TRACE_BITSTREAM( "  type = %u frameIndex = %u \n", (uint8_t)unitType, frameIndex );
  auto& sps = context.getSps();
  switch ( unitType ) {
    case PDG_PSPS: patchVpccParameterSet( pdg, bitstream ); break;
    case PDG_GPPS: geometryPatchParameterSet( pdg, bitstream ); break;
    case PDG_APPS: attributePatchParameterSet( pdg, sps, bitstream ); break;
    case PDG_PFPS: patchFrameParameterSet( pdg, sps, bitstream ); break;
    case PDG_PFAPS: patchFrameAttributeParameterSet( pdg, sps, bitstream ); break;
    case PDG_PFGPS: patchFrameGeometryParameterSet( pdg, sps, bitstream ); break;
    case PDG_PTGLU: patchTileGroupLayerUnit( pdg, frameIndex, context, bitstream ); break;
    case PDG_PREFIX_SEI: seiMessage( pdg, context, bitstream ); break;
    case PDG_SUFFIX_SEI: seiMessage( pdg, context, bitstream ); break;
    default: assert( 0 ); break;
  }
}

// 7.3.5.3 Patch sequence parameter set syntax
void PCCBitstreamDecoder::patchVpccParameterSet( PatchDataGroup& pdg, PCCBitstream& bitstream ) {
  TRACE_BITSTREAM( "%s \n", __func__ );
  uint32_t      index = bitstream.readUvlc();  // ue(v)
  auto&         psps  = pdg.getPatchVpccParameterSet( index );
  RefListStruct rls;
  psps.addRefListStruct( rls );
  psps.setPatchVpccParameterSetId( index );
  psps.setLog2PatchPackingBlockSize( bitstream.read( 3 ) );            // u(3)
  psps.setLog2MaxPatchFrameOrderCntLsbMinus4( bitstream.readUvlc() );  // ue(v)
  psps.setMaxDecPatchFrameBufferingMinus1( bitstream.readUvlc() );     // ue(v)
  psps.setLongTermRefPatchFramesFlag( bitstream.read( 1 ) );           // u(1)
  psps.setNumRefPatchFrameListsInPsps( bitstream.readUvlc() );         // ue(v)
  psps.allocate( psps.getNumRefPatchFrameListsInPsps() );
  for ( size_t i = 0; i < psps.getNumRefPatchFrameListsInPsps(); i++ ) {
    refListStruct( psps.getRefListStruct( i ), psps, bitstream );
  }
  psps.setUseEightOrientationsFlag( bitstream.read( 1 ) );                // u(1)
  psps.setNormalAxisLimitsQuantizationEnableFlag( bitstream.read( 1 ) );  // u(1)
  psps.setNormalAxisMaxDeltaValueEnableFlag( bitstream.read( 1 ) );       // u(1)
}

// 7.3.5.4 Patch frame geometry parameter set syntax
void PCCBitstreamDecoder::patchFrameGeometryParameterSet( PatchDataGroup&       pdg,
                                                          VpccParameterSet& sps,
                                                          PCCBitstream&         bitstream ) {
  TRACE_BITSTREAM( "%s \n", __func__ );
  size_t                       altasIndex              = 0;
  auto&    gi         = sps.getGeometryInformation(altasIndex);
  uint32_t pfgpsIndex = bitstream.readUvlc();  // ue(v)
  uint32_t pspsIndex  = bitstream.readUvlc();  // ue(v)
  auto&    pfgps      = pdg.getPatchFrameGeometryParameterSet( pfgpsIndex );
  pfgps.setPatchFrameGeometryParameterSetId( pfgpsIndex );
  pfgps.setPatchVpccParameterSetId( pspsIndex );
  TRACE_BITSTREAM( "GeometryParamsEnabledFlag = %d \n", gi.getGeometryParamsEnabledFlag() );
  if ( gi.getGeometryParamsEnabledFlag() ) { geometryFrameParams( pfgps.getGeometryFrameParams(), bitstream ); }
  TRACE_BITSTREAM( "GeometryPatchParamsEnabledFlag = %d \n", gi.getGeometryPatchParamsEnabledFlag() );
  if ( gi.getGeometryPatchParamsEnabledFlag() ) {
    pfgps.setGeometryPatchScaleParamsEnabledFlag( bitstream.read( 1 ) );     // u(1)
    pfgps.setGeometryPatchOffsetParamsEnabledFlag( bitstream.read( 1 ) );    // u(1)
    pfgps.setGeometryPatchRotationParamsEnabledFlag( bitstream.read( 1 ) );  // u(1)
    pfgps.setGeometryPatchPointSizeInfoEnabledFlag( bitstream.read( 1 ) );   // u(1)
    pfgps.setGeometryPatchPointShapeInfoEnabledFlag( bitstream.read( 1 ) );  // u(1)
  }
  byteAlignment( bitstream );
}

// 7.3.5.5 Geometry frame Params syntax
void PCCBitstreamDecoder::geometryFrameParams( GeometryFrameParams& gfp, PCCBitstream& bitstream ) {
  TRACE_BITSTREAM( "%s \n", __func__ );
  gfp.setGeometrySmoothingParamsPresentFlag( bitstream.read( 1 ) );  // u(1)
  gfp.setGeometryScaleParamsPresentFlag( bitstream.read( 1 ) );      // u(1)
  gfp.setGeometryOffsetParamsPresentFlag( bitstream.read( 1 ) );     // u(1)
  gfp.setGeometryRotationParamsPresentFlag( bitstream.read( 1 ) );   // u(1)
  gfp.setGeometryPointSizeInfoPresentFlag( bitstream.read( 1 ) );    // u(1)
  gfp.setGeometryPointShapeInfoPresentFlag( bitstream.read( 1 ) );   // u(1)
  if ( gfp.getGeometrySmoothingParamsPresentFlag() ) {
    gfp.setGeometrySmoothingEnabledFlag( bitstream.read( 1 ) );  // u(1)
    if ( gfp.getGeometrySmoothingEnabledFlag() ) {
      gfp.setGeometrySmoothingGridSizeMinus2( bitstream.read( 7 ) );  // u(7)
      gfp.setGeometrySmoothingThreshold( bitstream.read( 8 ) );       // u(8)
    }
    gfp.setGeometryPatchBlockFilteringEnableFlag( bitstream.read( 1 ) );  // u(1)
    if ( gfp.getGeometryPatchBlockFilteringEnableFlag() ) {
      gfp.setGeometryPatchBlockFilteringLog2ThresholdMinus1( bitstream.read( 2 ) );  // u(2)
      gfp.setGeometryPatchBlockFilteringPassesCountMinus1( bitstream.read( 2 ) );    // u(2)
      gfp.setGeometryPatchBlockFilteringFilterSizeMinus1( bitstream.read( 3 ) );     // u(3)
    }
  }
  if ( gfp.getGeometryScaleParamsPresentFlag() ) {
    for ( size_t d = 0; d < 3; d++ ) {
      gfp.setGeometryScaleOnAxis( d, bitstream.read( 32 ) );  // u(32)
    }
  }
  if ( gfp.getGeometryOffsetParamsPresentFlag() ) {
    for ( size_t d = 0; d < 3; d++ ) {
      gfp.setGeometryOffsetOnAxis( d, bitstream.readS( 32 ) );  // i(32)
    }
  }
  if ( gfp.getGeometryRotationParamsPresentFlag() ) {
    for ( size_t d = 0; d < 4; d++ ) {
      gfp.setGeometryRotationQuaternion(
          d, bitstream.readS( 32 ) );  // i(32) -> TODO: This should be i(16) according to the CD
    }
  }
  if ( gfp.getGeometryPointSizeInfoPresentFlag() ) {
    gfp.setGeometryPointSizeInfo( bitstream.read( 16 ) );  // u(16)
  }
  if ( gfp.getGeometryPointShapeInfoPresentFlag() ) {
    gfp.setGeometryPointShapeInfo( bitstream.read( 4 ) );  // u(4)
  }
}

// 7.3.5.6 Patch frame attribute parameter set syntax
void PCCBitstreamDecoder::patchFrameAttributeParameterSet( PatchDataGroup&       pdg,
                                                           VpccParameterSet& sps,
                                                           PCCBitstream&         bitstream ) {
  TRACE_BITSTREAM( "%s \n", __func__ );
  size_t atlasIndex = 0;
  auto&    ai         = sps.getAttributeInformation( atlasIndex );
  uint32_t pfapsIndex = bitstream.readUvlc();  // ue(v)
  uint32_t pspsIndex  = bitstream.readUvlc();  // ue(v)
  auto&    pfaps      = pdg.getPatchFrameAttributeParameterSet( pfapsIndex );
  // auto&    psps       = pdg.getPatchVpccParameterSet( pfapsIndex );
  pfaps.setPatchFrameAttributeParameterSetId( pfapsIndex );
  pfaps.setPatchSequencParameterSetId( pspsIndex );
  TRACE_BITSTREAM( "PatchFrameAttributeParameterSetId = %u  \n", pfaps.getPatchFrameAttributeParameterSetId() );
  TRACE_BITSTREAM( "PatchSequencParameterSetId       = %u  \n", pfaps.getPatchSequencParameterSetId() );
  size_t attributeDimension = ai.getAttributeDimensionMinus1( pfaps.getPatchFrameAttributeParameterSetId() ) + 1;
  TRACE_BITSTREAM( "attributeDimension = %lu \n", attributeDimension );
  if ( ai.getAttributeParamsEnabledFlag() ) {
    attributeFrameParams( pfaps.getAttributeFrameParams(), attributeDimension, bitstream );
  } else
    pfaps.getAttributeFrameParams().allocate( attributeDimension );

  if ( ai.getAttributePatchParamsEnabledFlag() ) {
    pfaps.setAttributePatchScaleParamsEnabledFlag( bitstream.read( 1 ) );   // u(1)
    pfaps.setAttributePatchOffsetParamsEnabledFlag( bitstream.read( 1 ) );  // u(1)
  }
  byteAlignment( bitstream );
}

// 7.3.5.7 Attribute frame Params syntax
void PCCBitstreamDecoder::attributeFrameParams( AttributeFrameParams& afp,
                                                size_t                attributeDimension,
                                                PCCBitstream&         bitstream ) {
  TRACE_BITSTREAM( "%s \n", __func__ );
  afp.allocate( attributeDimension );
  for ( size_t i = 0; i < attributeDimension; i++ ) {
    afp.setAttributeSmoothingParamsPresentFlag( i, bitstream.read( 1 ) );  // u(1)
  }
  afp.setAttributeScaleParamsPresentFlag( bitstream.read( 1 ) );   // u(1)
  afp.setAttributeOffsetParamsPresentFlag( bitstream.read( 1 ) );  // u(1)
  for ( size_t i = 0; i < attributeDimension; i++ ) {
    if ( afp.getAttributeSmoothingParamsPresentFlag( i ) ) {
      afp.setAttributeSmoothingGridSizeMinus2( i, bitstream.read( 8 ) );                // u(8)
      afp.setAttributeSmoothingThreshold( i, bitstream.read( 8 ) );                     // u(8)
      afp.setAttributeSmoothingLocalEntropyThreshold( i, bitstream.read( 3 ) );         // u(3)
      afp.setAttributeSmoothingThresholdAttributeVariation( i, bitstream.read( 8 ) );   // u(8)
      afp.setAttributeSmoothingThresholdAttributeDifference( i, bitstream.read( 8 ) );  // u(8)
    }
  }
  if ( afp.getAttributeScaleParamsPresentFlag() ) {
    for ( size_t i = 0; i < attributeDimension; i++ ) afp.setAttributeScale( i, bitstream.read( 32 ) );  // u32
  }
  if ( afp.getAttributeOffsetParamsPresentFlag() ) {
    for ( size_t i = 0; i < attributeDimension; i++ ) afp.setAttributeOffset( i, bitstream.readS( 32 ) );  // i32
  }
}

// 7.3.5.8 Geometry patch parameter set syntax
void PCCBitstreamDecoder::geometryPatchParameterSet( PatchDataGroup& pdg, PCCBitstream& bitstream ) {
  TRACE_BITSTREAM( "%s \n", __func__ );
  uint32_t gppsIndex  = bitstream.readUvlc();  // ue(v)
  uint32_t pfgpsIndex = bitstream.readUvlc();  // ue(v)
  auto&    gpps       = pdg.getGeometryPatchParameterSet( gppsIndex );
  auto&    pfgps      = pdg.getPatchFrameGeometryParameterSet( pfgpsIndex );
  gpps.setGeometryPatchParameterSetId( gppsIndex );
  gpps.setPatchFrameGeometryParameterSetId( pfgpsIndex );
  if ( pfgps.getGeometryPatchScaleParamsEnabledFlag() || pfgps.getGeometryPatchOffsetParamsEnabledFlag() ||
       pfgps.getGeometryPatchRotationParamsEnabledFlag() || pfgps.getGeometryPatchPointSizeInfoEnabledFlag() ||
       pfgps.getGeometryPatchPointShapeInfoEnabledFlag() ) {
    gpps.setGeometryPatchParamsPresentFlag( bitstream.read( 1 ) );  // u(1)
    if ( gpps.getGeometryPatchParamsPresentFlag() ) {
      geometryPatchParams( gpps.getGeometryPatchParams(), pfgps, bitstream );
    }
  }
  byteAlignment( bitstream );
}

// 7.3.5.9 Geometry patch Params syntax
void PCCBitstreamDecoder::geometryPatchParams( GeometryPatchParams&            gpp,
                                               PatchFrameGeometryParameterSet& gfps,
                                               PCCBitstream&                   bitstream ) {
  TRACE_BITSTREAM( "%s \n", __func__ );
  if ( gfps.getGeometryPatchScaleParamsEnabledFlag() ) {
    gpp.setGeometryPatchScaleParamsPresentFlag( bitstream.read( 1 ) );  // u(1)
    if ( gpp.getGeometryPatchScaleParamsPresentFlag() ) {
      for ( size_t d = 0; d < 3; d++ ) {
        gpp.setGeometryPatchScaleOnAxis( d, bitstream.read( 32 ) );  // u(32)
      }
    }
  }
  if ( gfps.getGeometryPatchOffsetParamsEnabledFlag() ) {
    gpp.setGeometryPatchOffsetParamsPresentFlag( bitstream.read( 1 ) );  // u(1)
    if ( gpp.getGeometryPatchOffsetParamsPresentFlag() ) {
      for ( size_t d = 0; d < 3; d++ ) {
        gpp.setGeometryPatchOffsetOnAxis( d, bitstream.readS( 32 ) );  // i(32)
      }
    }
  }
  if ( gfps.getGeometryPatchRotationParamsEnabledFlag() ) {
    gpp.setGeometryPatchRotationParamsPresentFlag( bitstream.read( 1 ) );  // u(1)
    if ( gpp.getGeometryPatchRotationParamsPresentFlag() ) {
      for ( size_t d = 0; d < 4; d++ ) {
        gpp.setGeometryPatchRotationQuaternion( d, bitstream.readS( 32 ) );  // i(32) -> TODO: i(16) according to the CD
      }
    }
  }
  if ( gfps.getGeometryPatchPointSizeInfoEnabledFlag() ) {
    gpp.setGeometryPatchPointSizeInfoPresentFlag( bitstream.read( 1 ) );  // u(1)
    if ( gpp.getGeometryPatchPointSizeInfoPresentFlag() ) {
      gpp.setGeometryPatchPointSizeInfo( bitstream.read( 16 ) );  // u(16)
    }
  }
  if ( gfps.getGeometryPatchPointShapeInfoEnabledFlag() ) {
    gpp.setGeometryPatchPointShapeInfoPresentFlag( bitstream.read( 1 ) );  // u(1)
    if ( gpp.getGeometryPatchPointShapeInfoPresentFlag() ) {
      gpp.setGeometryPatchPointShapeInfo( bitstream.read( 4 ) );  // u(4)
    }
  }
}

// 7.3.5.10 Attribute patch parameter set syntax
void PCCBitstreamDecoder::attributePatchParameterSet( PatchDataGroup&       pdg,
                                                      VpccParameterSet& sps,
                                                      PCCBitstream&         bitstream ) {
  TRACE_BITSTREAM( "%s \n", __func__ );
  // auto&    ai         = sps.getAttributeInformation();
  uint32_t appsIndex  = bitstream.readUvlc();  // ue(v)
  uint32_t pfapsIndex = bitstream.readUvlc();  // ue(v)
  auto&    apps       = pdg.getAttributePatchParameterSet( appsIndex );
  auto&    pfaps      = pdg.getPatchFrameAttributeParameterSet( pfapsIndex );
  apps.setAttributePatchParameterSetId( appsIndex );
  apps.setPatchFrameAttributeParameterSetId( pfapsIndex );
  apps.setAttributeDimensionMinus1( bitstream.read( 8 ) );
  size_t attributeDimension = apps.getAttributeDimensionMinus1() + 1;
  // size_t attributeDimension = ai.getAttributeDimensionMinus1( apps.getAttributePatchParameterSetId() ) + 1;
  if ( pfaps.getAttributePatchScaleParamsEnabledFlag() || pfaps.getAttributePatchOffsetParamsEnabledFlag() ) {
    apps.setAttributePatchParamsPresentFlag( bitstream.read( 1 ) );  // u(1)
    if ( apps.getAttributePatchParamsPresentFlag() ) {
      attributePatchParams( apps.getAttributePatchParams(), pfaps, attributeDimension, bitstream );
    }
  }
  byteAlignment( bitstream );
}

// 7.3.5.11 Attribute patch Params syntax
void PCCBitstreamDecoder::attributePatchParams( AttributePatchParams&            app,
                                                PatchFrameAttributeParameterSet& afps,
                                                size_t                           dimension,
                                                PCCBitstream&                    bitstream ) {
  TRACE_BITSTREAM( "%s \n", __func__ );
  if ( afps.getAttributePatchScaleParamsEnabledFlag() ) {
    app.setAttributePatchScaleParamsPresentFlag( bitstream.read( 1 ) );  // u(1)
    if ( app.getAttributePatchScaleParamsPresentFlag() ) {
      for ( size_t i = 0; i < dimension; i++ ) {
        app.setAttributePatchScale( i, bitstream.read( 32 ) );  // u(32)
      }
    }
  }
  if ( afps.getAttributePatchOffsetParamsEnabledFlag() ) {
    app.setAttributePatchOffsetParamsPresentFlag( bitstream.read( 1 ) );  // u(1)
    if ( app.getAttributePatchOffsetParamsPresentFlag() ) {
      for ( size_t i = 0; i < dimension; i++ ) {
        app.setAttributePatchOffset( i, bitstream.readS( 32 ) );  // i(32)
      }
    }
  }
}

// 7.3.5.12 Patch frame parameter set syntax
void PCCBitstreamDecoder::patchFrameParameterSet( PatchDataGroup&       pdg,
                                                  VpccParameterSet& sps,
                                                  PCCBitstream&         bitstream ) {
  TRACE_BITSTREAM( "%s \n", __func__ );
  size_t atlasIndex = 0;
  auto&    ai         = sps.getAttributeInformation(atlasIndex);
  uint32_t pfpsIndex  = bitstream.readUvlc();  // ue(v)
  uint32_t pspsIndex  = bitstream.readUvlc();  // ue(v)
  uint32_t gpfpsIndex = bitstream.readUvlc();  // ue(v)
  auto&    pfps       = pdg.getPatchFrameParameterSet( pfpsIndex );
  // auto&    psps      = pdg.getPatchVpccParameterSet( pspsIndex );
  pfps.setPatchFrameParameterSetId( pfpsIndex );
  pfps.setPatchVpccParameterSetId( pspsIndex );
  pfps.setGeometryPatchFrameParameterSetId( gpfpsIndex );
  TRACE_BITSTREAM( " ai.getAttributeCount() = %u \n", ai.getAttributeCount() );

  pfps.allocate( ai.getAttributeCount() );
  for ( size_t i = 0; i < ai.getAttributeCount(); i++ ) {
    pfps.setAttributePatchFrameParameterSetId( i, bitstream.readUvlc() );  // ue(v)
  }
  atlasFrameTileInformation( pfps.getAtlasFrameTileInformation(), sps, bitstream );

  pfps.setLocalOverrideGeometryPatchEnableFlag( bitstream.read( 1 ) );  // u(1)
  pfps.allocate( ai.getAttributeCount() );
  for ( size_t i = 0; i < ai.getAttributeCount(); i++ ) {
    pfps.setLocalOverrideAttributePatchEnableFlag( i, bitstream.read( 1 ) );  // u(1)
  }
  pfps.setAdditionalLtPfocLsbLen( bitstream.readUvlc() );  // ue(v)
  if ( sps.getProjection45DegreeEnableFlag() ) {
    pfps.setProjection45DegreeEnableFlag( bitstream.read( 1 ) );  // u(1)
  } else {
    pfps.setProjection45DegreeEnableFlag( false );
  }
  pfps.setLodModeEnableFlag(bitstream.read( 1 ) );  // u(1)
  byteAlignment( bitstream );
}


// 7.3.5.14 Patch tile group layer unit syntax
void PCCBitstreamDecoder::patchTileGroupLayerUnit( PatchDataGroup& pdg,
                                                   uint32_t        frameIndex,
                                                   PCCContext&     context,
                                                   PCCBitstream&   bitstream ) {
  TRACE_BITSTREAM( "%s \n", __func__ );
  auto& ptglu = pdg.addPatchTileGroupLayerUnit();
  auto& ptgh  = ptglu.getPatchTileGroupHeader();
  auto& ptgdu = ptglu.getPatchTileGroupDataUnit();
  auto& ptgluPrev =
      pdg.getPatchTileGroupLayerUnit( ( std::max )( 0, (int32_t)pdg.getPatchTileGroupLayerUnitSize() - 2 ) );
  ptglu.setFrameIndex( frameIndex );
  ptgh.setFrameIndex( frameIndex );
  ptgdu.setFrameIndex( frameIndex );
  patchTileGroupHeader( ptgh, ptgluPrev.getPatchTileGroupHeader(), context, bitstream );
  patchTileGroupDataUnit( ptgdu, ptgh, context, bitstream );
}

// 7.3.5.15 Patch tile group header syntax
void PCCBitstreamDecoder::patchTileGroupHeader( PatchTileGroupHeader& ptgh,
                                                PatchTileGroupHeader& pfhPrev,
                                                PCCContext&           context,
                                                PCCBitstream&         bitstream ) {
  TRACE_BITSTREAM( "%s \n", __func__ );
  auto&    sps        = context.getSps();
  size_t   atlasIndex = 0;
  auto&    gi         = sps.getGeometryInformation( atlasIndex );
  auto&    pdg        = context.getPatchDataGroup();
  uint32_t pfpsIndex  = bitstream.readUvlc();  // ue(v)
  ptgh.setPatchFrameParameterSetId( pfpsIndex );
  auto&   pfps     = pdg.getPatchFrameParameterSet( ptgh.getPatchFrameParameterSetId() );
  auto&   psps     = pdg.getPatchVpccParameterSet( pfps.getPatchVpccParameterSetId() );
  auto&   pfti     = pfps.getAtlasFrameTileInformation();
  uint8_t bitCount = pfti.getSignalledTileGroupIdLengthMinus1() + 1;
  ptgh.setAddress( bitstream.read( bitCount ) );
  // The length of ptgh_address is pfti_signalled_tile_group_id_length_minus1 + 1 bits.
  // If pfti_signalled_tile_group_id_flag is equal to 0, the value of ptgh_address shall be in the range of 0 to
  // pfti_num_tile_groups_in_patch_frame_minus1, inclusive. Otherwise, the value of ptgh_address shall be in the range
  // of 0 to 2( pfti_signalled_tile_group_id_length_minus1 + 1 ) − 1, inclusive.
  ptgh.setType( bitstream.readUvlc() );  // ue(v)
  bitCount = psps.getLog2MaxPatchFrameOrderCntLsbMinus4() + 4;
  ptgh.setPatchFrameOrderCntLsb( bitstream.read( bitCount ) );  // u(v)
  // The length of the ptgh_patch_frm_order_cnt_lsb syntax element is equal to
  // psps_log2_max_patch_frame_order_cnt_lsb_minus4 + 4 bits. The value of the ptgh_patch_frm_order_cnt_lsb shall be in
  // the range of 0 to MaxPatchFrmOrderCntLsb − 1, inclusive.

  TRACE_BITSTREAM( "Id     = %u \n", ptgh.getPatchFrameParameterSetId() );
  TRACE_BITSTREAM( "Adress = %u \n", ptgh.getAddress() );
  TRACE_BITSTREAM( "Type   = %u \n", ptgh.getType() );
  TRACE_BITSTREAM( "POC    = %u \n", ptgh.getPatchFrameOrderCntLsb() );
  TRACE_BITSTREAM( "psps.getNumRefPatchFrameListsInPsps() = %lu\n", psps.getNumRefPatchFrameListsInPsps() );
  TRACE_BITSTREAM( "psps.getNumRefPatchFrameListsInPsps() = %lu \n", psps.getNumRefPatchFrameListsInPsps() );
  TRACE_BITSTREAM( "gi.getGeometry3dCoordinatesBitdepthMinus1() = %lu \n",
                   gi.getGeometry3dCoordinatesBitdepthMinus1() );

  if ( psps.getNumRefPatchFrameListsInPsps() > 0 ) {
    ptgh.setRefPatchFrameListSpsFlag( bitstream.read( 1 ) );  // u( 1 )
  }
  if ( ptgh.getRefPatchFrameListSpsFlag() ) {
    if ( psps.getNumRefPatchFrameListsInPsps() > 1 ) {
      bitCount = getFixedLengthCodeBitsCount( psps.getNumRefPatchFrameListsInPsps() + 1 );
      ptgh.setRefPatchFrameListIdx(
          bitstream.read( bitCount ) );  // u( v ) :Ceil( Log2( psps_num_ref_patch_frame_lists_in_psps ) )
    }
  } else {
    auto& rls = psps.addRefListStruct();
    refListStruct( rls, psps, bitstream );
  }
  uint8_t rlsIdx =
      psps.getNumRefPatchFrameListsInPsps() ? ptgh.getRefPatchFrameListIdx() : psps.getNumRefPatchFrameListsInPsps();
  size_t numLtrpEntries = 0;
  for ( size_t i = 0; i < psps.getRefListStruct( rlsIdx ).getNumRefEntries(); i++ ) {
    if ( !psps.getRefListStruct( rlsIdx ).getStRefAtalsFrameFlag( i ) ) { numLtrpEntries++; }
  }

  for ( size_t j = 0; j < numLtrpEntries; j++ ) {
    ptgh.setAdditionalPfocLsbPresentFlag( j, bitstream.read( 1 ) );  // u( 1 )
    if ( ptgh.getAdditionalPfocLsbPresentFlag( j ) ) {
      uint8_t bitCount = pfps.getAdditionalLtPfocLsbLen();
      ptgh.setAdditionalPfocLsbVal( j, bitstream.read( bitCount ) );  // u(v) : pfps_additional_lt_pfoc_lsb_len
    }
  }

  ptgh.setNormalAxisMinValueQuantizer( 0 );
  ptgh.setNormalAxisMaxDeltaValueQuantizer( 0 );
  if ( psps.getNormalAxisLimitsQuantizationEnableFlag() ) {
    ptgh.setNormalAxisMinValueQuantizer( bitstream.read( 5 ) );
    if ( psps.getNormalAxisMaxDeltaValueEnableFlag() ) ptgh.setNormalAxisMaxDeltaValueQuantizer( bitstream.read( 5 ) );
  }

  const uint8_t maxBitCountForMinDepth = uint8_t( gi.getGeometry3dCoordinatesBitdepthMinus1() );
  const uint8_t maxBitCountForMaxDepth = uint8_t( gi.getGeometry3dCoordinatesBitdepthMinus1() );
  // min
  ptgh.setInterPredictPatch3dShiftNormalAxisBitCountMinus1( maxBitCountForMinDepth );

  // max
  if ( pfps.getProjection45DegreeEnableFlag() == 0 ) {
    ptgh.setInterPredictPatch2dDeltaSizeDBitCountMinus1( maxBitCountForMaxDepth );
  } else {
    ptgh.setInterPredictPatch2dDeltaSizeDBitCountMinus1( maxBitCountForMaxDepth + 1 );
  }

  if ( ptgh.getType() == PATCH_FRAME_P && psps.getRefListStruct( rlsIdx ).getNumRefEntries() > 1 ) {
    ptgh.setNumRefIdxActiveOverrideFlag( bitstream.read( 1 ) );                                              // u( 1 )
    if ( ptgh.getNumRefIdxActiveOverrideFlag() ) { ptgh.setNumRefIdxActiveMinus1( bitstream.readUvlc() ); }  // ue(v)
  }

  if ( ptgh.getType() == PATCH_FRAME_I ) {
    ptgh.setInterPredictPatch2dShiftUBitCountMinus1( bitstream.read( 8 ) );              // u( 8 )
    ptgh.setInterPredictPatch2dShiftVBitCountMinus1( bitstream.read( 8 ) );              // u( 8 )
    ptgh.setInterPredictPatch3dShiftTangentAxisBitCountMinus1( bitstream.read( 8 ) );    // u( 8 )
    ptgh.setInterPredictPatch3dShiftBitangentAxisBitCountMinus1( bitstream.read( 8 ) );  // u( 8 )
    ptgh.setInterPredictPatchLodBitCount( bitstream.read( 8 ) );                         // u( 8 )
  } else {
    ptgh.setInterPredictPatchBitCountFlag( bitstream.read( 1 ) );  // u( 1 )
    if ( ptgh.getInterPredictPatchBitCountFlag() ) {
      ptgh.setInterPredictPatch2dShiftUBitCountFlag( bitstream.read( 1 ) );  // u( 1 )
      if ( ptgh.getInterPredictPatch2dShiftUBitCountFlag() ) {
        ptgh.setInterPredictPatch2dShiftUBitCountMinus1( bitstream.read( 8 ) );
      }                                                                      // u( 8 )
      ptgh.setInterPredictPatch2dShiftVBitCountFlag( bitstream.read( 1 ) );  // u( 1 )
      if ( ptgh.getInterPredictPatch2dShiftVBitCountFlag() ) {
        ptgh.setInterPredictPatch2dShiftVBitCountMinus1( bitstream.read( 8 ) );
      }                                                                                // u( 8 )
      ptgh.setInterPredictPatch3dShiftTangentAxisBitCountFlag( bitstream.read( 1 ) );  // u( 1 )
      if ( ptgh.getInterPredictPatch3dShiftTangentAxisBitCountFlag() ) {
        ptgh.setInterPredictPatch3dShiftTangentAxisBitCountMinus1( bitstream.read( 8 ) );  // u( 8 )
      }
      ptgh.setInterPredictPatch3dShiftBitangentAxisBitCountFlag( bitstream.read( 1 ) );  // u( 1 )
      if ( ptgh.getInterPredictPatch3dShiftBitangentAxisBitCountFlag() ) {
        ptgh.setInterPredictPatch3dShiftBitangentAxisBitCountMinus1( bitstream.read( 8 ) );  // u( 8 )
      }
      ptgh.setInterPredictPatchLodBitCountFlag( bitstream.read( 1 ) );  // u( 1 )
      if ( ptgh.getInterPredictPatchLodBitCountFlag() ) {
        ptgh.setInterPredictPatchLodBitCount( bitstream.read( 8 ) + 1 );
      }  // u( 8 )
    }
    if ( !ptgh.getInterPredictPatchBitCountFlag() || !ptgh.getInterPredictPatch2dShiftUBitCountFlag() ) {
      ptgh.setInterPredictPatch2dShiftUBitCountMinus1( pfhPrev.getInterPredictPatch2dShiftUBitCountMinus1() );
    }
    if ( !ptgh.getInterPredictPatchBitCountFlag() || !ptgh.getInterPredictPatch2dShiftVBitCountFlag() ) {
      ptgh.setInterPredictPatch2dShiftVBitCountMinus1( pfhPrev.getInterPredictPatch2dShiftVBitCountMinus1() );
    }
    if ( !ptgh.getInterPredictPatchBitCountFlag() || !ptgh.getInterPredictPatch3dShiftTangentAxisBitCountFlag() ) {
      ptgh.setInterPredictPatch3dShiftTangentAxisBitCountMinus1(
          pfhPrev.getInterPredictPatch3dShiftTangentAxisBitCountMinus1() );
    }
    if ( !ptgh.getInterPredictPatchBitCountFlag() || !ptgh.getInterPredictPatch3dShiftBitangentAxisBitCountFlag() ) {
      ptgh.setInterPredictPatch3dShiftBitangentAxisBitCountMinus1(
          pfhPrev.getInterPredictPatch3dShiftBitangentAxisBitCountMinus1() );
    }
    if ( !ptgh.getInterPredictPatchBitCountFlag() || !ptgh.getInterPredictPatchLodBitCountFlag() ) {
      ptgh.setInterPredictPatchLodBitCount( pfhPrev.getInterPredictPatchLodBitCount() );
    }
  }
  if ( sps.getEnhancedOccupancyMapForDepthFlag() && sps.getEOMTexturePatch() ) {
    ptgh.setEOMPatchNbPatchBitCountMinus1( bitstream.read( 8 ) );  // u( 8 )
    ptgh.setEOMPatchMaxEPBitCountMinus1( bitstream.read( 8 ) );    // u( 8 )
  }
  TRACE_BITSTREAM("RawPatchEnabledFlag = %d \n", sps.getRawPatchEnabledFlag(atlasIndex) );
  if ( sps.getRawPatchEnabledFlag(atlasIndex) ) {
  // sps_pcm_patch_enabled_flag
    ptgh.setRaw3dShiftBitCountPresentFlag( bitstream.read( 1 ) );  // u( 1 )
    if ( ptgh.getRaw3dShiftBitCountPresentFlag() ) {
      ptgh.setRaw3dShiftAxisBitCountMinus1( bitstream.read( gi.getGeometry3dCoordinatesBitdepthMinus1() + 1 ) );  //
    }
  } else {
    size_t bitCountPcmU1V1D1 = gi.getGeometry3dCoordinatesBitdepthMinus1() - gi.getGeometryNominal2dBitdepthMinus1();
    ptgh.setRaw3dShiftAxisBitCountMinus1( bitCountPcmU1V1D1 - 1 );
  }

  TRACE_BITSTREAM(
      "InterPredictPatchBitCount Flag %d %d %d %d %d %d %d Count = %u %u %u %u %u %u \n",
      ptgh.getInterPredictPatchBitCountFlag(), ptgh.getInterPredictPatch2dShiftUBitCountFlag(),
      ptgh.getInterPredictPatch2dShiftVBitCountFlag(), ptgh.getInterPredictPatch3dShiftTangentAxisBitCountFlag(),
      ptgh.getInterPredictPatch3dShiftBitangentAxisBitCountFlag(), ptgh.getInterPredictPatchLodBitCountFlag(),
      ptgh.getRaw3dShiftBitCountPresentFlag(), ptgh.getInterPredictPatch2dShiftUBitCountMinus1(),
      ptgh.getInterPredictPatch2dShiftVBitCountMinus1(), ptgh.getInterPredictPatch3dShiftTangentAxisBitCountMinus1(),
      ptgh.getInterPredictPatch3dShiftBitangentAxisBitCountMinus1(), ptgh.getInterPredictPatchLodBitCount(),
      ptgh.getRaw3dShiftAxisBitCountMinus1() );
  byteAlignment( bitstream );
}

// 7.3.5.16 Reference list structure syntax (OLD)
void PCCBitstreamDecoder::refListStruct( RefListStruct&             rls,
                                         PatchVpccParameterSet& psps,
                                         PCCBitstream&              bitstream ) {
  TRACE_BITSTREAM( "%s \n", __func__ );
  rls.setNumRefEntries( bitstream.readUvlc() );  // ue(v)
  TRACE_BITSTREAM( "NumRefEntries = %lu  \n", rls.getNumRefEntries() );
  rls.allocate();
  for ( size_t i = 0; i < rls.getNumRefEntries(); i++ ) {
    if ( psps.getLongTermRefPatchFramesFlag() ) {
      rls.setStRefAtalsFrameFlag( i, bitstream.read( 1 ) );  // u(1)
      if ( rls.getStRefAtalsFrameFlag( i ) ) {
        rls.setAbsDeltaPfocSt( i, bitstream.readUvlc() );  // ue(v)
        if ( rls.getAbsDeltaPfocSt( i ) > 0 ) {
          rls.setStrpfEntrySignFlag( i, bitstream.read( 1 ) );  // u(1)
        } else {
          uint8_t bitCount = psps.getLog2MaxPatchFrameOrderCntLsbMinus4() + 4;
          rls.setAfocLsbLt(
              i, bitstream.read( bitCount ) );  // u(v) : psps_log2_max_patch_frame_order_cnt_lsb_minus4  + 4 bits
        }
      }
    }
  }
  TRACE_BITSTREAM( "%s done \n", __func__ );
}

// 7.3.5.16 Reference list structure syntax (NEW jkei: moved up)
//void PCCBitstreamDecoder::refListStruct( RefListStruct&             rls,
//                                         AtlasSequenceParameterSetRBSP& asps,
//                                         PCCBitstream&              bitstream ) {
//  TRACE_BITSTREAM( "%s \n", __func__ );
//  rls.setNumRefEntries( bitstream.readUvlc() );  // ue(v)
//  TRACE_BITSTREAM( "NumRefEntries = %lu  \n", rls.getNumRefEntries() );
//  rls.allocate();
//  for ( size_t i = 0; i < rls.getNumRefEntries(); i++ ) {
//    if ( asps.getLongTermRefAtlasFramesFlag() ) {
//      rls.setStRefAtalsFrameFlag( i, bitstream.read( 1 ) );  // u(1)
//      if ( rls.getStRefAtalsFrameFlag( i ) ) {
//        rls.setAbsDeltaPfocSt( i, bitstream.readUvlc() );  // ue(v)
//        if ( rls.getAbsDeltaPfocSt( i ) > 0 ) {
//          rls.setStrpfEntrySignFlag( i, bitstream.read( 1 ) );  // u(1)
//        } else {
//          uint8_t bitCount = asps.getLog2MaxAtlasFrameOrderCntLsbMinus4() + 4;
//          rls.setAfocLsbLt(
//              i, bitstream.read( bitCount ) );  // u(v) : psps_log2_max_patch_frame_order_cnt_lsb_minus4  + 4 bits
//        }
//      }
//    }
//  }
//  TRACE_BITSTREAM( "%s done \n", __func__ );
//}

// 7.3.6.1 General patch tile group data unit syntax
void PCCBitstreamDecoder::patchTileGroupDataUnit( PatchTileGroupDataUnit& ptgdu,
                                                  PatchTileGroupHeader&   ptgh,
                                                  PCCContext&             context,
                                                  PCCBitstream&           bitstream ) {
  TRACE_BITSTREAM( "%s \n", __func__ );
  TRACE_BITSTREAM( "ptgh.getType()        = %lu \n", ptgh.getType() );
  ptgdu.init();
  prevPatchSizeU_   = 0;
  prevPatchSizeV_   = 0;
  predPatchIndex_   = 0;
  size_t patchIndex = 0;

  PCCPatchFrameType tileGroupType = (PCCPatchFrameType)ptgh.getType();
  uint8_t           patchMode     = bitstream.readUvlc();  // ue(v)
  TRACE_BITSTREAM( "patchMode = %lu \n", patchMode );
  while ( !( ( ( tileGroupType == PATCH_FRAME_I ) && ( patchMode == PATCH_MODE_I_END ) ) ||
             ( ( tileGroupType == PATCH_FRAME_P ) && ( patchMode == PATCH_MODE_P_END ) ) ) ) {
    auto& pid = ptgdu.addPatchInformationData( patchMode );
    pid.setFrameIndex( ptgdu.getFrameIndex() );
    pid.setPatchIndex( patchIndex );
    patchIndex++;
    patchInformationData( pid, patchMode, ptgh, context, bitstream );
    patchMode = bitstream.readUvlc();  // ue(v)
    TRACE_BITSTREAM( "patchMode = %lu \n", patchMode );
  }
  TRACE_BITSTREAM( "ptgdu.getPatchCount() = %lu \n", ptgdu.getPatchCount() );
  byteAlignment( bitstream );
}

// 7.3.6.2 Patch information data syntax
void PCCBitstreamDecoder::patchInformationData( PatchInformationData& pid,
                                                size_t                patchMode,
                                                PatchTileGroupHeader& ptgh,
                                                PCCContext&           context,
                                                PCCBitstream&         bitstream ) {
  TRACE_BITSTREAM( "%s \n", __func__ );
  auto& sps  = context.getSps();
  size_t atlasIndex = 0;
  auto& ai   = sps.getAttributeInformation(atlasIndex);
  auto& pdg  = context.getPatchDataGroup();
  auto& pfps = pdg.getPatchFrameParameterSet( ptgh.getPatchFrameParameterSetId() );
  pid.allocate( ai.getAttributeCount() );
  if ( ( PCCPatchFrameType( ptgh.getType() ) ) == PATCH_FRAME_P && patchMode == PATCH_MODE_P_SKIP ) {
    // skip mode: currently not supported but added it for convenience. Could easily be removed
  } else if ( ( ( PCCPatchFrameType( ptgh.getType() ) ) == PATCH_FRAME_I && patchMode == PATCH_MODE_I_INTRA ) ||
              ( ( PCCPatchFrameType( ptgh.getType() ) ) == PATCH_FRAME_P && patchMode == PATCH_MODE_P_INTRA ) ) {
    if ( pfps.getLocalOverrideGeometryPatchEnableFlag() ) {
      pid.setOverrideGeometryPatchFlag( bitstream.read( 1 ) );  // u(1)
      if ( pid.getOverrideGeometryPatchFlag() ) {
        pid.setGeometryPatchParameterSetId( bitstream.readUvlc() );  // ue(v)
        TRACE_BITSTREAM( " gppsId = %lu \n", pid.getGeometryPatchParameterSetId() );
      }
    }
    TRACE_BITSTREAM( " ai.getAttributeCount() = %lu \n", ai.getAttributeCount() );
    pfps.allocate( ai.getAttributeCount() );
    for ( int i = 0; i < ai.getAttributeCount(); i++ ) {
      TRACE_BITSTREAM( " override flag = %lu \n", pfps.getLocalOverrideAttributePatchEnableFlag( i ) );
      if ( pfps.getLocalOverrideAttributePatchEnableFlag( i ) ) {
        const bool overrideAttributePatchFlag = bitstream.read( 1 );  // u(1)
        TRACE_BITSTREAM( " overrideAttributePatchFlag = %lu \n", overrideAttributePatchFlag );
        pid.addOverrideAttributePatchFlag( overrideAttributePatchFlag );
      } else {
        pid.addOverrideAttributePatchFlag( false );
      }
      TRACE_BITSTREAM( " override attribute patch flag = %lu \n", pid.getOverrideAttributePatchFlag( i ) );
      if ( pid.getOverrideAttributePatchFlag( i ) ) {
        pid.addAttributePatchParameterSetId( bitstream.readUvlc() );  // ue(v)
        TRACE_BITSTREAM( " AttributePatchParameterSetId = %lu \n", pid.getAttributePatchParameterSetId( i ) );
      } else {
        pid.addAttributePatchParameterSetId( 0 );
      }
    }
    auto& pdu = pid.getPatchDataUnit();
    pdu.setFrameIndex( pid.getFrameIndex() );
    pdu.setPatchIndex( pid.getPatchIndex() );
    patchDataUnit( pdu, ptgh, context, bitstream );
  } else if ( ( PCCPatchFrameType( ptgh.getType() ) ) == PATCH_FRAME_P && patchMode == PATCH_MODE_P_INTER ) {
    auto& dpdu = pid.getDeltaPatchDataUnit();
    dpdu.setDpduFrameIndex( pid.getFrameIndex() );
    dpdu.setDpduPatchIndex( pid.getPatchIndex() );
    deltaPatchDataUnit( dpdu, ptgh, context, bitstream );
  } else if ( ( ( PCCPatchFrameType( ptgh.getType() ) ) == PATCH_FRAME_I && patchMode == PATCH_MODE_I_Raw ) ||
              ( ( PCCPatchFrameType( ptgh.getType() ) ) == PATCH_FRAME_P && patchMode == PATCH_MODE_P_Raw ) ) {
    auto& ppdu = pid.getRawPatchDataUnit();
    ppdu.setFrameIndex( pid.getFrameIndex() );
    ppdu.setPatchIndex( pid.getPatchIndex() );
    pcmPatchDataUnit( ppdu, ptgh, context, bitstream );
  } else if ( ( ( PCCPatchFrameType( ptgh.getType() ) ) == PATCH_FRAME_I && patchMode == PATCH_MODE_I_EOM ) ||
              ( ( PCCPatchFrameType( ptgh.getType() ) ) == PATCH_FRAME_P && patchMode == PATCH_MODE_P_EOM ) ) {
    auto& epdu = pid.getEomPatchDataUnit();
    eomPatchDataUnit( epdu, ptgh, context, bitstream );
  }
}

// 7.3.6.3 Patch data unit syntax
void PCCBitstreamDecoder::patchDataUnit( PatchDataUnit&        pdu,
                                         PatchTileGroupHeader& ptgh,
                                         PCCContext&           context,
                                         PCCBitstream&         bitstream ) {
  auto& sps                          = context.getSps();
  auto  ptghPatchFrameParameterSetId = ptgh.getPatchFrameParameterSetId();
  auto& pfps = context.getPatchDataGroup().getPatchFrameParameterSet( ptghPatchFrameParameterSetId );
  auto  pfpsPatchVpccParameterSetId = pfps.getPatchVpccParameterSetId();
  auto& psps = context.getPatchDataGroup().getPatchVpccParameterSet( pfpsPatchVpccParameterSetId );

  TRACE_BITSTREAM( "%s \n", __func__ );
  pdu.setPdu2dPosX( bitstream.read( ptgh.getInterPredictPatch2dShiftUBitCountMinus1() + 1 ) );  // u(v)
  pdu.setPdu2dPosY( bitstream.read( ptgh.getInterPredictPatch2dShiftVBitCountMinus1() + 1 ) );  // u(v)
  pdu.setPdu2dDeltaSizeX( bitstream.readSvlc() );                                                 // se(v)
  pdu.setPdu2dDeltaSizeY( bitstream.readSvlc() );                                                 // se(v)
  pdu.setPdu3dPosX(
      bitstream.read( ptgh.getInterPredictPatch3dShiftTangentAxisBitCountMinus1() + 1 ) );  // u(v)
  pdu.setPdu3dPosY(
      bitstream.read( ptgh.getInterPredictPatch3dShiftBitangentAxisBitCountMinus1() + 1 ) );  // u(v)
  pdu.setPdu3dPosMinZ(
      bitstream.read( ptgh.getInterPredictPatch3dShiftNormalAxisBitCountMinus1() + 1 ) );  // u(v)
  if ( psps.getNormalAxisMaxDeltaValueEnableFlag() ) {
    pdu.setPdu3dPosDeltaMaxZ(
        bitstream.read( ptgh.getInterPredictPatch2dDeltaSizeDBitCountMinus1() + 1 ) );  // u(v)
  }
  pdu.setProjectPlane( bitstream.read( 3 ) );
  if ( psps.getUseEightOrientationsFlag() ) {
    pdu.setPduOrientationIndex( bitstream.read( 3 ) );  // u(3)
  } else {
    pdu.setPduOrientationIndex( bitstream.read( 1 ) );  // u(1)
  }
  if ( pfps.getLodModeEnableFlag()) {
    pdu.setLodEnableFlag(bitstream.read( 1 ) ); //u1
    if( pdu.getLodEnableFlag() ) {
      pdu.setLodScaleXminus1( uint8_t( bitstream.readUvlc()) ); //uev
      pdu.setLodScaleY( uint8_t( bitstream.readUvlc()) ); //uev
    }
  }
  // auto& pfps = pdg.getPatchFrameParameterSet( 0 );
  if ( pfps.getProjection45DegreeEnableFlag() ) {
    pdu.set45DegreeProjectionPresentFlag( bitstream.read( 1 ) );  // u(1)
  }
  if ( pdu.get45DegreeProjectionPresentFlag() ) {
    pdu.set45DegreeProjectionRotationAxis( bitstream.read( 2 ) );  // u(2)
  } else {
    pdu.set45DegreeProjectionRotationAxis( 0 );
  }
  if ( sps.getPointLocalReconstructionEnabledFlag() ) {
    auto& plrd = pdu.getPointLocalReconstructionData();
    plrd.allocate( prevPatchSizeU_ + pdu.getPdu2dDeltaSizeX(), prevPatchSizeV_ + pdu.getPdu2dDeltaSizeY() );
    pointLocalReconstructionData( plrd, context, bitstream );
    prevPatchSizeU_ += pdu.getPdu2dDeltaSizeX();
    prevPatchSizeV_ += pdu.getPdu2dDeltaSizeY();
  }
  TRACE_BITSTREAM( "Patch(%zu/%zu) => UV %4lu %4lu S=%4ld %4ld P=%zu O=%d A=%lu %lu %lu P45= %d %d lod=(%lu) %lu %lu\n ",
                   pdu.getPatchIndex(), pdu.getFrameIndex(), pdu.getPdu2dPosX(), pdu.getPdu2dPosY(),
                   pdu.getPdu2dDeltaSizeX(), pdu.getPdu2dDeltaSizeY(), (size_t)pdu.getProjectPlane(),
                   pdu.getPduOrientationIndex(), pdu.getPdu3dPosX(), pdu.getPdu3dPosY(),
                   pdu.getPdu3dPosMinZ(), pdu.get45DegreeProjectionPresentFlag(),
                   pdu.get45DegreeProjectionRotationAxis(),
                   pdu.getLodEnableFlag(),pdu.getLodScaleXminus1(),pdu.getLodScaleY()
                  );
}

// 7.3.6.4  Delta Patch data unit syntax
void PCCBitstreamDecoder::deltaPatchDataUnit( DeltaPatchDataUnit&   dpdu,
                                              PatchTileGroupHeader& ptgh,
                                              PCCContext&           context,
                                              PCCBitstream&         bitstream ) {
  auto& sps = context.getSps();

  auto  ptghPatchFrameParameterSetId = ptgh.getPatchFrameParameterSetId();
  auto& pfps = context.getPatchDataGroup().getPatchFrameParameterSet( ptghPatchFrameParameterSetId );
  auto  pfpsPatchVpccParameterSetId = pfps.getPatchVpccParameterSetId();
  auto& psps = context.getPatchDataGroup().getPatchVpccParameterSet( pfpsPatchVpccParameterSetId );

  TRACE_BITSTREAM( "%s \n", __func__ );
  dpdu.setDeltaPatchIdx( bitstream.readSvlc() );              // se(v)
  dpdu.set2DDeltaShiftU( bitstream.readSvlc() );              // se(v)
  dpdu.set2DDeltaShiftV( bitstream.readSvlc() );              // se(v)
  dpdu.set2DDeltaSizeU( bitstream.readSvlc() );               // se(v)
  dpdu.set2DDeltaSizeV( bitstream.readSvlc() );               // se(v)
  dpdu.set3DDeltaShiftTangentAxis( bitstream.readSvlc() );    // se(v)
  dpdu.set3DDeltaShiftBiTangentAxis( bitstream.readSvlc() );  // se(v)
  dpdu.set3DDeltaShiftMinNormalAxis( bitstream.readSvlc() );  // se(v)
  if ( psps.getNormalAxisMaxDeltaValueEnableFlag() ) {
    dpdu.set3DShiftDeltaMaxNormalAxis( bitstream.readSvlc() );  // se(v)
  }
  if ( sps.getPointLocalReconstructionEnabledFlag() ) {
    auto&  plrd      = dpdu.getPointLocalReconstructionData();
    auto&  pdg       = context.getPatchDataGroup();
    auto&  pfluPrev  = pdg.getPatchTileGroupLayerUnit( predFramePatchTileGroupLayerUnitIndex_ );
    auto&  pfhPrev   = pfluPrev.getPatchTileGroupHeader();
    auto&  pfduPrev  = pfluPrev.getPatchTileGroupDataUnit();
    auto&  pidPrev   = pfduPrev.getPatchInformationData( dpdu.getDeltaPatchIdx() + predPatchIndex_ );
    auto   patchMode = pfduPrev.getPatchMode( dpdu.getDeltaPatchIdx() + predPatchIndex_ );
    size_t sizeU     = dpdu.get2DDeltaSizeU();
    size_t sizeV     = dpdu.get2DDeltaSizeV();
    if ( ( ( PCCPatchFrameType( pfhPrev.getType() ) ) == PATCH_FRAME_I && patchMode == PATCH_MODE_I_INTRA ) ||
         ( ( PCCPatchFrameType( pfhPrev.getType() ) ) == PATCH_FRAME_P && patchMode == PATCH_MODE_P_INTRA ) ) {
      sizeU += pidPrev.getPatchDataUnit().getPointLocalReconstructionData().getBlockToPatchMapWidth();
      sizeV += pidPrev.getPatchDataUnit().getPointLocalReconstructionData().getBlockToPatchMapHeight();
    } else if ( ( PCCPatchFrameType( pfhPrev.getType() ) ) == PATCH_FRAME_P && patchMode == PATCH_MODE_P_INTER ) {
      sizeU += pidPrev.getDeltaPatchDataUnit().getPointLocalReconstructionData().getBlockToPatchMapWidth();
      sizeV += pidPrev.getDeltaPatchDataUnit().getPointLocalReconstructionData().getBlockToPatchMapHeight();
    }
    plrd.allocate( sizeU, sizeV );
    pointLocalReconstructionData( plrd, context, bitstream );
    prevPatchSizeU_ = sizeU;
    prevPatchSizeV_ = sizeV;
    predPatchIndex_ += ( dpdu.getDeltaPatchIdx() + 1 );
  }

  TRACE_BITSTREAM(
      "%zu frame %zu DeltaPatch => DeltaIdx = %d ShiftUV = %ld %ld DeltaSize = %ld %ld Axis = %ld %ld %ld\n",
      dpdu.getDpduFrameIndex(), dpdu.getDpduPatchIndex(), dpdu.getDeltaPatchIdx(), dpdu.get2DDeltaShiftU(),
      dpdu.get2DDeltaShiftV(), dpdu.get2DDeltaSizeU(), dpdu.get2DDeltaSizeV(), dpdu.get3DDeltaShiftTangentAxis(),
      dpdu.get3DDeltaShiftBiTangentAxis(), dpdu.get3DDeltaShiftMinNormalAxis() );
}

 //7.3.6.5 raw patch data unit syntax (jkei: moved up)
void PCCBitstreamDecoder::pcmPatchDataUnit( RawPatchDataUnit&     ppdu,
                                            PatchTileGroupHeader& ptgh,
                                            PCCContext&           context,
                                            PCCBitstream&         bitstream ) {
  auto& sps = context.getSps();
  TRACE_BITSTREAM( "%s \n", __func__ );
  size_t atlasIndex = 0;
  if ( sps.getRawSeparateVideoPresentFlag(atlasIndex) ) { ppdu.setRpduPatchInRawVideoFlag( bitstream.read( 1 ) ); }  // u(1)
  ppdu.setRpdu2dPosX( bitstream.read( ptgh.getInterPredictPatch2dShiftUBitCountMinus1() + 1 ) );         // u(v)
  ppdu.setRpdu2dPosY( bitstream.read( ptgh.getInterPredictPatch2dShiftVBitCountMinus1() + 1 ) );         // u(v)
  ppdu.setRpdu2dDeltaSizeX( bitstream.readSvlc() );                                                        // se(v)
  ppdu.setRpdu2dDeltaSizeY( bitstream.readSvlc() );                                                        // se(v)
  ppdu.setRpdu3dPosX( bitstream.read( ptgh.getRaw3dShiftAxisBitCountMinus1() + 1 ) );          // u(v)
  ppdu.setRpdu3dPosY( bitstream.read( ptgh.getRaw3dShiftAxisBitCountMinus1() + 1 ) );        // u(v)
  ppdu.setRpdu3dPosZ( bitstream.read( ptgh.getRaw3dShiftAxisBitCountMinus1() + 1 ) );           // u(v)
  ppdu.setRpduRawPoints( bitstream.readUvlc() );
  TRACE_BITSTREAM(
      "Raw Patch => UV %4lu %4lu  S=%4ld %4ld  UVD1=%4ld %4ld %4ld NumPcmPoints=%lu PatchInRawVideoFlag=%d \n",
      ppdu.getRpdu2dPosX(), ppdu.getRpdu2dPosY(), ppdu.getRpdu2dDeltaSizeX(), ppdu.getRpdu2dDeltaSizeY(),
      ppdu.getRpdu3dPosX(), ppdu.getRpdu3dPosY(), ppdu.getRpdu3dPosZ(), ppdu.getRpduRawPoints(),
      ppdu.getRpduPatchInRawVideoFlag() );
}

// 7.3.6.x EOM patch data unit syntax
void PCCBitstreamDecoder::eomPatchDataUnit( EOMPatchDataUnit&     epdu,
                                            PatchTileGroupHeader& ptgh,
                                            PCCContext&           context,
                                            PCCBitstream&         bitstream ) {
  TRACE_BITSTREAM( "%s \n", __func__ );
  epdu.setEpdu2dPosX( bitstream.read( ptgh.getInterPredictPatch2dShiftUBitCountMinus1() + 1 ) );  // u(v)
  epdu.setEpdu2dPosY( bitstream.read( ptgh.getInterPredictPatch2dShiftVBitCountMinus1() + 1 ) );  // u(v)
  epdu.setEpdu2dDeltaSizeX( bitstream.readSvlc() );                                                 // se(v)
  epdu.setEpdu2dDeltaSizeY( bitstream.readSvlc() );                                                 // se(v)
  epdu.setEpduAssociatedPatchesCountMinus1( bitstream.read( 8) );
  for(size_t cnt=0; cnt<epdu.getEpduAssociatedPatchesCountMinus1()+1; cnt++)
  {
    epdu.setEpduAssociatedPatches( bitstream.read( 8 ), cnt);
    epdu.setEpduEomPointsPerPatch( bitstream.readUvlc(), cnt);  //???
  }
  TRACE_BITSTREAM( "EOM Patch => UV %4lu %4lu  N=%4ld\n", epdu.getEpdu2dPosX(),
                  epdu.getEpdu2dPosY(), epdu.getEpduAssociatedPatchesCountMinus1()+1);
  for(size_t cnt=0; cnt<epdu.getEpduAssociatedPatchesCountMinus1()+1; cnt++)
    TRACE_BITSTREAM( "%4ld, %4ld\n", epdu.getEpduAssociatedPatches()[cnt], epdu.getEpduEomPointsPerPatch()[cnt]);
}

// 7.3.6.6 Point local reconstruction data syntax
void PCCBitstreamDecoder::pointLocalReconstructionData( PointLocalReconstructionData& plrd,
                                                        PCCContext&                   context,
                                                        PCCBitstream&                 bitstream ) {
  auto& plri = context.getSps().getPointLocalReconstructionInformation();
  TRACE_BITSTREAM( "%s \n", __func__ );
  TRACE_BITSTREAM( "WxH= %lu x %lu \n", plrd.getBlockToPatchMapWidth(), plrd.getBlockToPatchMapHeight() );

  const size_t  blockCount   = plrd.getBlockToPatchMapWidth() * plrd.getBlockToPatchMapHeight();
  const uint8_t bitCountMode = uint8_t( getFixedLengthCodeBitsCount( uint32_t( plri.getNumberOfModesMinus1() ) ) );
  TRACE_BITSTREAM( "  bitCountMode = %u \n", bitCountMode );

  if ( blockCount > plri.getBlockThresholdPerPatchMinus1() + 1 ) {
    plrd.setLevelFlag( bitstream.read( 1 ) );
  } else {
    plrd.setLevelFlag( true );
  }
  TRACE_BITSTREAM( "  LevelFlag = %u \n", plrd.getLevelFlag() );

  if ( plrd.getLevelFlag() ) {
    plrd.setPresentFlag( bitstream.read( 1 ) );
    if ( plrd.getPresentFlag() ) { plrd.setModeMinus1( bitstream.read( bitCountMode ) ); }
    TRACE_BITSTREAM( "  ModePatch: Present = %d ModeMinus1 = %d \n", plrd.getPresentFlag(),
                     plrd.getPresentFlag() ? (int32_t)plrd.getModeMinus1() : -1 );
  } else {
    for ( size_t i = 0; i < blockCount; i++ ) {
      plrd.setBlockPresentFlag( i, bitstream.read( 1 ) );
      if ( plrd.getBlockPresentFlag( i ) ) { plrd.setBlockModeMinus1( i, bitstream.read( bitCountMode ) ); }
      TRACE_BITSTREAM( "  Mode[ %4lu / %4lu ]: Present = %d ModeMinus1 = %d \n", i, blockCount,
                       plrd.getBlockPresentFlag( i ),
                       plrd.getBlockPresentFlag( i ) ? plrd.getBlockModeMinus1( i ) : -1 );
    }
  }
#ifdef BITSTREAM_TRACE
  for ( size_t v0 = 0; v0 < plrd.getBlockToPatchMapHeight(); ++v0 ) {
    for ( size_t u0 = 0; u0 < plrd.getBlockToPatchMapWidth(); ++u0 ) {
      size_t i = v0 * plrd.getBlockToPatchMapWidth() + u0;
      TRACE_BITSTREAM( "Patch Block[ %2lu %2lu <=> %4lu ] / [ %2lu %2lu ] Level = %d Present = %d Mode = %d \n", u0, v0,
                       i, plrd.getBlockToPatchMapWidth(), plrd.getBlockToPatchMapHeight(), plrd.getLevelFlag(),
                       plrd.getLevelFlag() ? plrd.getPresentFlag() : plrd.getBlockPresentFlag( i ),
                       plrd.getLevelFlag()
                           ? plrd.getPresentFlag() ? (int32_t)plrd.getModeMinus1() : -1
                           : plrd.getBlockPresentFlag( i ) ? plrd.getBlockModeMinus1( i ) : -1 );
    }
  }
#endif
}

// 7.3.5.22 Supplemental enhancement information message syntax TODO: Implement
void PCCBitstreamDecoder::seiMessage( PatchDataGroup& pdg, PCCContext& context, PCCBitstream& bitstream ) {}


// JR TODO: continue

// 7.3.5 NAL unit syntax
// 7.3.5.1 General NAL unit syntax
void PCCBitstreamDecoder::nalUnit( PCCBitstream& bitstream, NalUnit& nalUnit ) {
  nalUnitHeader( bitstream, nalUnit );
  for( size_t i = 0; i < nalUnit.getNalUnitSize() - 2; i++ ) {
    nalUnit.setNalUnitData( i, bitstream.read( 8 ) ); // b(8)
  }
}

// 7.3.5.2 NAL unit header syntax
void PCCBitstreamDecoder::nalUnitHeader( PCCBitstream& bitstream, NalUnit& nalUnit ) {
  bitstream.read( 1 ); // f(1)
  nalUnit.setUnitType(  bitstream.read( 6 ) );// u(6)
  nalUnit.setLayerId( bitstream.read( 6 ) ); // u(6)
  nalUnit.setTemporalyIdPlus1( bitstream.read( 3 ) ); // u(3)
}

// C.2 Sample stream NAL unit syntax and semantics
// C.2.1 Sample stream NAL header syntax
void PCCBitstreamDecoder::sampleStreamNalHeader(  PCCBitstream& bitstream, SampleStreamNalUnit& sampleStreamNalUnit ) {
  sampleStreamNalUnit.setUnitSizePrecisionBytesMinus1( bitstream.read( 3 ) ); // u(3)
  bitstream.read( 5 ); // u(5)  
}

// C.2.2 Sample stream NAL unit syntax
void PCCBitstreamDecoder::sampleStreamNalUnit( PCCBitstream& bitstream, SampleStreamNalUnit& sampleStreamNalUnit ) {
  auto &nalu = sampleStreamNalUnit.addNalUnit();
  nalu.setNalUnitSize( bitstream.read( sampleStreamNalUnit.getUnitSizePrecisionBytesMinus1() + 1 ) ); // u(v)  
  nalu.allocate();
  nalUnit( bitstream, nalu );
} 

// F.2.1 VUI parameters syntax
void PCCBitstreamDecoder::vuiParameters( ) {
  // vui_timing_info_present_flag //   u(1)
  // if( vui_timing_info_present_flag ) {
  //   vui_num_units_in_tick // u(32)
  //   vui_time_scale // u(32)
  //   vui_poc_proportional_to_timing_flag // u(1)
  //   if( vui_poc_proportional_to_timing_flag ) {
  //     vui_num_ticks_poc_diff_one_minus1 // ue(v)
  //   }
  //   vui_hrd_parameters_present_flag   // u(1)
  //   if( vui_hrd_parameters_present_flag ){
  //     hrdParameters()
  //   }
  // }
}
// F.2.2 HRD parameters syntax
void PCCBitstreamDecoder::hrdParameters( ) {
  // hrd_nal_parameters_present_flag // u(1)
  // hrd_acl_parameters_present_flag // u(1)
  // if( hrd_nal_parameters_present_flag | | hrd_acl_parameters_present_flag ){
  //   hrd_bit_rate_scale // u(4)
  //   hrd_cab_size_scale // u(4)
  //   hrd_initial_cab_removal_delay_length_minus1 // u(5)
  //   hrd_au_cab_removal_delay_length_minus1 // u(5)
  //   hrd_dab_output_delay_length_minus1  // u(5)
  // }
  // maxNumSubLayersMinus1 = 0
  // for( i = 0; i <= maxNumSubLayersMinus1; i++ ) {
  //   hrd_fixed_atlas_rate_general_flag[ i ] // u(1)
  // if( !hrd_fixed_atlas_rate_general_flag[ i ] )
  //   hrd_fixed_atlas_rate_within_cas_flag[ i ] // u(1)
  // if( hrd_fixed_atlas_rate_within_cas_flag[ i ] )
  //   hrd_elemental_duration_in_tc_minus1[ i ] // ue(v)
  // else
  //   hrd_low_delay_flag[ i ] // u(1)
  // if( !hrd_low_delay_flag[ i ] )
  //   hrd_cab_cnt_minus1[ i ] // ue(v)
  // if( hrd_nal_parameters_present_flag )
  //   hrd_sub_layer_parameters( 0, i )
  // if( hrd_acl_parameters_present_flag )
  //   hrd_sub_layer_parameters( 1, i )
  // }
}

// F.2.3 Sub-layer HRD parameters syntax
void PCCBitstreamDecoder::hrdSubLayerParameters() {
  // for( i = 0; i <= CabCnt; i++ ) {
  //  hrd_bit_rate_value_minus1[ type ][ i ] // ue(v)
  //  hrd_cab_size_value_minus1[ type ][ i ]  // ue(v)
  //  hrd_cbr_flag[ type ][ i ] // u(1)
  // }
}

// JR TODO: remove 
// 7.3.2.3 raw separate video data syntax - TODO: remove this
void PCCBitstreamDecoder::pcmSeparateVideoData( PCCContext& context, PCCBitstream& bitstream, uint8_t bitCount ) {
  TRACE_BITSTREAM( "%s \n", __func__ );
  auto& vpcc = context.getVPCC();
  auto& sps  = context.getSps();
  size_t atlasIndex = 0;
  if ( sps.getRawSeparateVideoPresentFlag( atlasIndex ) && !vpcc.getMapIndex() ) {
    vpcc.setRawVideoFlag( bitstream.read( 1 ) );  // u(1)
    bitstream.read( bitCount );                   // u(bitCount)
  } else {
    bitstream.read( bitCount + 1 );  // u(bitCount + 1)
  }
}
