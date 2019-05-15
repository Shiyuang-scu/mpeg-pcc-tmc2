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
#include "PCCBitstream.h"
#include "PCCContext.h"
#include "PCCFrameContext.h"
#include "PCCPatch.h"
#include "PCCGroupOfFrames.h"
#include "PCCEncoderParameters.h"
#include <tbb/tbb.h>
#include "PCCChrono.h"

#include "PCCBitstreamEncoder.h"

using namespace std;
using namespace pcc;

PCCBitstreamEncoder::PCCBitstreamEncoder() {}
PCCBitstreamEncoder::~PCCBitstreamEncoder() {}

void PCCBitstreamEncoder::setParameters( PCCEncoderParameters params ) { params_ = params; }

int PCCBitstreamEncoder::encode( PCCContext& context, PCCBitstream& bitstream ) {
  bitstream.getBitStreamStat().newGOF();
  vpccUnit( context, bitstream, VPCC_SPS );
  vpccUnit( context, bitstream, VPCC_PSD );
  vpccUnit( context, bitstream, VPCC_OVD );
  vpccUnit( context, bitstream, VPCC_GVD );
  vpccUnit( context, bitstream, VPCC_AVD );
  std::cout << " occupancy map  ->" << bitstream.getBitStreamStat().getTotalMetadata() << " B " << std::endl;
  return 0;
}

void PCCBitstreamEncoder::vpccVideoDataUnit( PCCContext& context, PCCBitstream& bitstream, VPCCUnitType vpccUnitType ) {
  TRACE_BITSTREAM( "%s \n", __func__ );
  auto& sps = context.getSps();
  if ( vpccUnitType == VPCC_OVD ) {
    TRACE_BITSTREAM( "OccupancyMap \n" );
    bitstream.write( context.getVideoBitstream( VIDEO_OCCUPANCY ) );
  } else if ( vpccUnitType == VPCC_GVD ) {
    TRACE_BITSTREAM( "Geometry \n" );
    if ( !sps.getLayerAbsoluteCodingEnabledFlag( 1 ) ) {
      bitstream.write( context.getVideoBitstream( VIDEO_GEOMETRY_D0 ) );
      bitstream.write( context.getVideoBitstream( VIDEO_GEOMETRY_D1 ) );
    } else {
      bitstream.write( context.getVideoBitstream( VIDEO_GEOMETRY ) );
    }
    if ( sps.getPcmPatchEnabledFlag() && sps.getPcmSeparateVideoPresentFlag() ) {
      bitstream.write( context.getVideoBitstream( VIDEO_GEOMETRY_MP ) );
    }
  } else if ( vpccUnitType == VPCC_AVD ) {
    if ( sps.getAttributeInformation().getAttributeCount() > 0 ) {
      TRACE_BITSTREAM( "Texture \n" );
      bitstream.write( context.getVideoBitstream( VIDEO_TEXTURE ) );
      if ( sps.getPcmPatchEnabledFlag() && sps.getPcmSeparateVideoPresentFlag() ) {
        bitstream.write( context.getVideoBitstream( VIDEO_TEXTURE_MP ) );
      }
    }
  }
}

// 7.3.2.1 General V-PCC unit syntax
void PCCBitstreamEncoder::vpccUnit( PCCContext& context, PCCBitstream& bitstream, VPCCUnitType vpccUnitType ) {
  TRACE_BITSTREAM( "%s \n", __func__ );
  int32_t position = bitstream.size();
  vpccUnitHeader( context, bitstream, vpccUnitType );
  vpccUnitPayload( context, bitstream, vpccUnitType );
  // while( more_data_in_vpcc_unit() ) { bitstream.write( 0, 8 ); }
  bitstream.getBitStreamStat().setVpccUnitSize( vpccUnitType, (int32_t)bitstream.size() - position );
}

// 7.3.2.2 V-PCC unit header syntax TODO: write attributeDimensionIndex_, modify pcmSeparateVideoData parameters for
// VPCC_AVD case
void PCCBitstreamEncoder::vpccUnitHeader( PCCContext& context, PCCBitstream& bitstream, VPCCUnitType vpccUnitType ) {
  TRACE_BITSTREAM( "%s \n", __func__ );
  auto& vpcc = context.getVPCC();
  auto& sps  = context.getSps();
  bitstream.write( vpccUnitType, 5 );  // u(5)
  if ( vpccUnitType == VPCC_AVD || vpccUnitType == VPCC_GVD || vpccUnitType == VPCC_OVD || vpccUnitType == VPCC_PSD ) {
    bitstream.write( (uint32_t)vpcc.getSequenceParameterSetId(), 4 );  // u(4)
  }
  if ( vpccUnitType == VPCC_AVD ) {
    bitstream.write( (uint32_t)vpcc.getAttributeIndex(), 7 );           // u(7)
    bitstream.write( (uint32_t)vpcc.getAttributeDimensionIndex(), 7 );  // u(7)
    if ( sps.getMultipleLayerStreamsPresentFlag() ) {
      bitstream.write( (uint32_t)vpcc.getLayerIndex(), 4 );  // u(4)
      pcmSeparateVideoData( context, bitstream, 4 );
    } else {
      pcmSeparateVideoData( context, bitstream, 8 );
    }
  } else if ( vpccUnitType == VPCC_GVD ) {
    if ( sps.getMultipleLayerStreamsPresentFlag() ) {
      bitstream.write( vpcc.getLayerIndex(), 4 );  // u(4)
      pcmSeparateVideoData( context, bitstream, 18 );
    } else {
      pcmSeparateVideoData( context, bitstream, 22 );
    }
  } else if ( vpccUnitType == VPCC_OVD || vpccUnitType == VPCC_PSD ) {
    bitstream.write( (uint32_t)0, 23 );  // u(23)
  } else {
    bitstream.write( (uint32_t)0, 27 );  // u(27)
  }
}

// 7.3.2.3 PCM separate video data syntax
void PCCBitstreamEncoder::pcmSeparateVideoData( PCCContext& context, PCCBitstream& bitstream, uint8_t bitCount ) {
  TRACE_BITSTREAM( "%s \n", __func__ );
  auto& vpcc = context.getVPCC();
  auto& sps  = context.getSps();
  if ( sps.getPcmSeparateVideoPresentFlag() && !vpcc.getLayerIndex() ) {
    bitstream.write( (uint32_t)vpcc.getPCMVideoFlag(), 1 );  // u(1)
    bitstream.write( (uint32_t)0, bitCount );                // u(bitCount)
  } else {
    bitstream.write( (uint32_t)0, bitCount + 1 );  // u(bitCount + 1)
  }
}

// 7.3.2.4 V-PCC unit payload syntax
void PCCBitstreamEncoder::vpccUnitPayload( PCCContext& context, PCCBitstream& bitstream, VPCCUnitType vpccUnitType ) {
  TRACE_BITSTREAM( "%s \n", __func__ );
  auto& sps = context.getSps();
  TRACE_BITSTREAM( "vpccUnitType = %d \n", (int32_t)vpccUnitType );
  if ( vpccUnitType == VPCC_SPS ) {
    sequenceParameterSet( sps, context, bitstream );
  } else if ( vpccUnitType == VPCC_PSD ) {
    patchDataGroup( context, bitstream );
  } else if ( vpccUnitType == VPCC_OVD || vpccUnitType == VPCC_GVD || vpccUnitType == VPCC_AVD ) {
    vpccVideoDataUnit( context, bitstream, vpccUnitType );
  }
}

// 7.3.3 Byte alignment syntax
void PCCBitstreamEncoder::byteAlignment( PCCBitstream& bitstream ) {
  TRACE_BITSTREAM( "%s \n", __func__ );
  bitstream.write( 1, 1 );  // f(1): equal to 1
  while ( !bitstream.byteAligned() ) {
    bitstream.write( 0, 1 );  // f(1): equal to 0
  }
}

// 7.3.4.1 General Sequence parameter set syntax TODO: remove sps.setAttributeCount() and loop over attributes
void PCCBitstreamEncoder::sequenceParameterSet( SequenceParameterSet& sps,
                                                PCCContext&           context,
                                                PCCBitstream&         bitstream ) {
  TRACE_BITSTREAM( "%s \n", __func__ );
  profileTierLevel( sps.getProfileTierLevel(), bitstream );
  bitstream.write( (uint32_t)sps.getSequenceParameterSetId(), 4 );   // u(4)
  bitstream.write( (uint32_t)sps.getFrameWidth(), 16 );              // u(16)
  bitstream.write( (uint32_t)sps.getFrameHeight(), 16 );             // u(16)
  bitstream.write( (uint32_t)sps.getAvgFrameRatePresentFlag(), 1 );  // u(1)
  if ( sps.getAvgFrameRatePresentFlag() ) {
    bitstream.write( (uint32_t)sps.getAvgFrameRate(), 16 );  // u(16)
  }
  bitstream.write( (uint32_t)sps.getEnhancedOccupancyMapForDepthFlag(), 1 );  // u(1)
  bitstream.write( (uint32_t)sps.getLayerCountMinus1(), 4 );                  // u(4)
  if ( sps.getLayerCountMinus1() > 0 ) {
    bitstream.write( (uint32_t)sps.getMultipleLayerStreamsPresentFlag(), 1 );  // u(1)
  }
  for ( size_t i = 0; i < sps.getLayerCountMinus1(); i++ ) {
    bitstream.write( (uint32_t)sps.getLayerAbsoluteCodingEnabledFlag( i + 1 ), 1 );  // u(1)
    if ( ( sps.getLayerAbsoluteCodingEnabledFlag( i + 1 ) == 0 ) ) {
      if ( i > 0 ) {
        bitstream.writeUvlc( (uint32_t)sps.getLayerPredictorIndexDiff( i + 1 ) );  // ue(v)
      }
    }
  }
  TRACE_BITSTREAM( " LayerCountMinus1  = %lu \n", sps.getLayerCountMinus1() );
  TRACE_BITSTREAM( " AbsoluteCoding L0 = %lu \n", sps.getLayerAbsoluteCodingEnabledFlag( 0 ) );
  TRACE_BITSTREAM( " AbsoluteCoding L1 = %lu \n", sps.getLayerAbsoluteCodingEnabledFlag( 1 ) );
  bitstream.write( (uint32_t)sps.getPcmPatchEnabledFlag(), 1 );  // u(1)
  if ( sps.getPcmPatchEnabledFlag() ) {
    bitstream.write( (uint32_t)sps.getPcmSeparateVideoPresentFlag(), 1 );  // u(1)
  }
  occupancyInformation( sps.getOccupancyInformation(), bitstream );
  geometryInformation( sps.getGeometryInformation(), sps, bitstream );
  attributeInformation( sps.getAttributeInformation(), sps, bitstream );

  bitstream.write( (uint32_t)sps.getPatchInterPredictionEnabledFlag(), 1 );      // u(1)
  bitstream.write( (uint32_t)sps.getPixelDeinterleavingFlag(), 1 );              // u(1)
  bitstream.write( (uint32_t)sps.getPointLocalReconstructionEnabledFlag(), 1 );  // u(1)
  if ( sps.getPointLocalReconstructionEnabledFlag() ) {
    pointLocalReconstructionInformation( sps.getPointLocalReconstructionInformation(), context, bitstream );
  }
  bitstream.write( (uint32_t)sps.getRemoveDuplicatePointEnabledFlag(), 1 );  // u(1)
  bitstream.write( (uint32_t)sps.getProjection45DegreeEnableFlag(), 1 );     // u(1)
  bitstream.write( (uint32_t)sps.getPatchPrecedenceOrderFlag(), 1 );         // u(1)

  // THE NEXT PARAMETERS ARE NOT IN THE VPCC CD SYNTAX DOCUMENTS AND WILL BE REMOVE
  bitstream.write( (uint32_t)sps.getLosslessGeo444(), 1 );    // u(1)
  bitstream.write( (uint32_t)sps.getLosslessGeo(), 1 );       // u(1)
  bitstream.write( (uint32_t)sps.getLosslessTexture(), 1 );   // u(1)
  bitstream.write( (uint32_t)sps.getMinLevel(), 8 );          // u(8)
  bitstream.write( (uint32_t)sps.getSurfaceThickness(), 8 );  // u(8)
  // THE NEXT PARAMETERS ARE NOT IN THE VPCC CD SYNTAX DOCUMENTS AND WILL BE REMOVE

  byteAlignment( bitstream );
}

// 7.3.4.2 Profile, tier, and level syntax TODO: change profileIdc to profileCodecGroupIdc, missing
// profilePCCToolsetIdc, profileReconstructionIdc
void PCCBitstreamEncoder::profileTierLevel( ProfileTierLevel& ptl, PCCBitstream& bitstream ) {
  TRACE_BITSTREAM( "%s \n", __func__ );
  bitstream.write( (uint32_t)ptl.getTierFlag(), 1 );                  // u(1)
  bitstream.write( (uint32_t)ptl.getProfileCodecGroupIdc(), 7 );      // u(7)
  bitstream.write( (uint32_t)ptl.getProfilePccToolsetIdc(), 8 );      // u(8)
  bitstream.write( (uint32_t)ptl.getProfileReconctructionIdc(), 8 );  // u(8)
  bitstream.write( (uint32_t)0, 32 );                                 // u(32)
  bitstream.write( (uint32_t)ptl.getLevelIdc(), 8 );                  // u(8)
}

// 7.3.4.3 Occupancy parameter set syntax
void PCCBitstreamEncoder::occupancyInformation( OccupancyInformation& oi, PCCBitstream& bitstream ) {
  TRACE_BITSTREAM( "%s \n", __func__ );
  bitstream.write( (uint32_t)oi.getOccupancyCodecId(), 8 );                       // u(8)
  bitstream.write( (uint32_t)oi.getLossyOccupancyMapCompressionThreshold(), 8 );  // u(8)
  TRACE_BITSTREAM( "  OccupancyLossyThreshold = %d  \n", oi.getLossyOccupancyMapCompressionThreshold() );
}

// 7.3.4.4 Geometry parameter set syntax
void PCCBitstreamEncoder::geometryInformation( GeometryInformation&  gi,
                                               SequenceParameterSet& sps,
                                               PCCBitstream&         bitstream ) {
  TRACE_BITSTREAM( "%s \n", __func__ );
  bitstream.write( (uint32_t)gi.getGeometryCodecId(), 8 );                            // u(8)
  bitstream.write( (uint32_t)gi.getGeometryNominal2dBitdepthMinus1(), 5 );            // u(5)
  bitstream.write( ( uint32_t )( gi.getGeometry3dCoordinatesBitdepthMinus1() ), 5 );  // u(5)
  if ( sps.getPcmSeparateVideoPresentFlag() ) {
    bitstream.write( (uint32_t)gi.getPcmGeometryCodecId(), 8 );  // u(8)
  }
  bitstream.write( (uint32_t)gi.getGeometryParamsEnabledFlag(), 1 );       // u(1)
  bitstream.write( (uint32_t)gi.getGeometryPatchParamsEnabledFlag(), 1 );  // u(1)
  TRACE_BITSTREAM( "GeometryParamsEnabledFlag = %d \n", gi.getGeometryParamsEnabledFlag() );
  TRACE_BITSTREAM( "GeometryPatchParamsEnabledFlag = %d \n", gi.getGeometryPatchParamsEnabledFlag() );
}

// 7.3.4.5 Attribute information
void PCCBitstreamEncoder::attributeInformation( AttributeInformation& ai,
                                                SequenceParameterSet& sps,
                                                PCCBitstream&         bitstream ) {
  TRACE_BITSTREAM( "%s \n", __func__ );
  bitstream.write( (uint32_t)ai.getAttributeCount(), 7 );  // u(7)
  TRACE_BITSTREAM( "AttributeCount = %u  \n", ai.getAttributeCount() );
  for ( size_t i = 0; i < ai.getAttributeCount(); i++ ) {
    bitstream.write( (uint32_t)ai.getAttributeTypeId( i ), 4 );   // u(4)
    bitstream.write( (uint32_t)ai.getAttributeCodecId( i ), 8 );  // u(8)
    if ( sps.getPcmSeparateVideoPresentFlag() ) {
      bitstream.write( (uint32_t)ai.getPcmAttributeCodecId( i ), 8 );  // u(8)
    }
    bitstream.write( (uint32_t)ai.getAttributeDimensionMinus1( i ), 8 );  // u(8)
    if ( ai.getAttributeDimensionMinus1( i ) > 0 ) {
      bitstream.write( (uint32_t)ai.getAttributeDimensionPartitionsMinus1( i ), 7 );  // u(7)
      int32_t remainingDimensions = ai.getAttributeDimensionMinus1( i );
      int32_t k                   = ai.getAttributeDimensionPartitionsMinus1( i );
      for ( int32_t j = 0; j < k; j++ ) {
        if ( k - j != remainingDimensions ) {
          bitstream.writeUvlc( (uint32_t)ai.getAttributePartitionChannelsMinus1( i, j ) );  // ue(v)
        }
        remainingDimensions -= ai.getAttributePartitionChannelsMinus1( i, j ) + 1;
      }
    }
    bitstream.write( (uint32_t)ai.getAttributeNominal2dBitdepthMinus1( i ), 5 );  // u(5)
  }
  if ( ai.getAttributeCount() > 0 ) {
    bitstream.write( (uint32_t)ai.getAttributeParamsEnabledFlag(), 1 );       // u(1)
    bitstream.write( (uint32_t)ai.getAttributePatchParamsEnabledFlag(), 1 );  // u(1)
    bitstream.write( (uint32_t)ai.getAttributeMSBAlignFlag(), 1 );            // u(1)
  }
}

// 7.3.5.1 General patch data group unit syntax TODO: rename(?)
void PCCBitstreamEncoder::patchDataGroup( PCCContext& context, PCCBitstream& bitstream ) {
  TRACE_BITSTREAM( "%s \n", __func__ );
  size_t prevPFLUindex = 0;
  auto&  pdg           = context.getPatchDataGroup();
  size_t frameCount    = 0;
  for ( uint8_t unitType = (uint8_t)PDG_PSPS; unitType < (uint8_t)PDG_PTGLU; unitType++ ) {
    bitstream.writeUvlc( ( (uint32_t)unitType ) );  // ue(v)
    TRACE_BITSTREAM( "%s (%u): frame = %u \n", strUnitType( (PDGUnitType)unitType ).c_str(), (uint32_t)unitType, 0 );
    patchDataGroupUnitPayload( pdg, (PDGUnitType)unitType, 0, 0, context, bitstream );
    bitstream.write( (uint32_t)0, 1 );  // u(1)
  }
  for ( uint8_t i = 0; i < pdg.getPatchTileGroupLayerUnitSize(); i++ ) {
    bitstream.writeUvlc( ( (uint32_t)PDG_PTGLU ) );  // ue(v)
    TRACE_BITSTREAM( "%s (%u): frame = %u \n", strUnitType( PDG_PTGLU ).c_str(), (uint32_t)PDG_PTGLU, i );
    patchDataGroupUnitPayload( pdg, PDG_PTGLU, i, i, context, bitstream );
    bitstream.write( ( uint32_t )( i + 1 == pdg.getPatchTileGroupLayerUnitSize() ), 1 );  // u(1)
  }
  byteAlignment( bitstream );
}

// 7.3.5.2 Patch data group unit payload syntax
void PCCBitstreamEncoder::patchDataGroupUnitPayload( PatchDataGroup& pdg,
                                                     PDGUnitType     unitType,
                                                     size_t          index,
                                                     size_t          frameIndex,
                                                     PCCContext&     context,
                                                     PCCBitstream&   bitstream ) {
  TRACE_BITSTREAM( "%s \n", __func__ );
  TRACE_BITSTREAM( "  type = %u frameIndex = %u \n", (uint8_t)unitType, frameIndex );
  auto& sps = context.getSps();
  switch ( unitType ) {
    case PDG_PSPS: patchSequenceParameterSet( pdg, index, bitstream ); break;
    case PDG_GPPS: geometryPatchParameterSet( pdg, index, bitstream ); break;
    case PDG_APPS: attributePatchParameterSet( pdg, index, sps, bitstream ); break;
    case PDG_PFPS: patchFrameParameterSet( pdg, index, sps, bitstream ); break;
    case PDG_PFAPS: patchFrameAttributeParameterSet( pdg, index, sps, bitstream ); break;
    case PDG_PFGPS: patchFrameGeometryParameterSet( pdg, index, sps, bitstream ); break;
    case PDG_PTGLU: patchTileGroupLayerUnit( pdg, index, context, bitstream ); break;
  }
}

// 7.3.5.3 Patch sequence parameter set syntax
void PCCBitstreamEncoder::patchSequenceParameterSet( PatchDataGroup& pdg, size_t index, PCCBitstream& bitstream ) {
  TRACE_BITSTREAM( "%s \n", __func__ );
  auto& psps = pdg.getPatchSequenceParameterSet( index );
  bitstream.writeUvlc( psps.getPatchSequenceParameterSetId() );         // ue(v)
  bitstream.write( psps.getLog2PatchPackingBlockSize(), 3 );            // u(1)
  bitstream.writeUvlc( psps.getLog2MaxPatchFrameOrderCntLsbMinus4() );  // ue(v)
  bitstream.writeUvlc( psps.getMaxDecPatchFrameBufferingMinus1() );     // ue(v)
  bitstream.write( psps.getLongTermRefPatchFramesFlag(), 1 );           // u(1)
  bitstream.writeUvlc( psps.getNumRefPatchFrameListsInPsps() );         // ue(v)
  for ( size_t i = 0; i < psps.getNumRefPatchFrameListsInPsps(); i++ ) {
    refListStruct( psps.getRefListStruct( i ), psps, bitstream );
  }
  bitstream.write( psps.getUseEightOrientationsFlag(), 1 );                // u(1)
  bitstream.write( psps.getNormalAxisLimitsQuantizationEnableFlag(), 1 );  // u(1)
  bitstream.write( psps.getNormalAxisMaxDeltaValueEnableFlag(), 1 );       // u(1)
}

// 7.3.5.4 Patch frame geometry parameter set syntax
void PCCBitstreamEncoder::patchFrameGeometryParameterSet( PatchDataGroup&       pdg,
                                                          size_t                index,
                                                          SequenceParameterSet& sps,
                                                          PCCBitstream&         bitstream ) {
  TRACE_BITSTREAM( "%s \n", __func__ );
  auto& gi    = sps.getGeometryInformation();
  auto& pfgps = pdg.getPatchFrameGeometryParameterSet( index );
  auto& psps  = pdg.getPatchSequenceParameterSet( pfgps.getPatchSequenceParameterSetId() );
  bitstream.writeUvlc( pfgps.getPatchFrameGeometryParameterSetId() );  // ue(v)
  bitstream.writeUvlc( pfgps.getPatchSequenceParameterSetId() );       // ue(v)
  TRACE_BITSTREAM( "GeometryParamsEnabledFlag = %d \n", gi.getGeometryParamsEnabledFlag() );
  if ( gi.getGeometryParamsEnabledFlag() ) { geometryFrameParams( pfgps.getGeometryFrameParams(), bitstream ); }
  TRACE_BITSTREAM( "GeometryPatchParamsEnabledFlag = %d \n", gi.getGeometryPatchParamsEnabledFlag() );
  if ( gi.getGeometryPatchParamsEnabledFlag() ) {
    bitstream.write( (uint32_t)pfgps.getOverrideGeometryPatchParamsFlag(), 1 );  // u(1)
    if ( pfgps.getOverrideGeometryPatchParamsFlag() ) {
      bitstream.write( (uint32_t)pfgps.getGeometryPatchScaleParamsEnabledFlag(), 1 );     // u(1)
      bitstream.write( (uint32_t)pfgps.getGeometryPatchOffsetParamsEnabledFlag(), 1 );    // u(1)
      bitstream.write( (uint32_t)pfgps.getGeometryPatchRotationParamsEnabledFlag(), 1 );  // u(1)
      bitstream.write( (uint32_t)pfgps.getGeometryPatchPointSizeInfoEnabledFlag(), 1 );   // u(1)
      bitstream.write( (uint32_t)pfgps.getGeometryPatchPointShapeInfoEnabledFlag(), 1 );  // u(1)
    }
  }
  byteAlignment( bitstream );
}

// 7.3.5.5 Geometry frame Params syntax
void PCCBitstreamEncoder::geometryFrameParams( GeometryFrameParams& gfp, PCCBitstream& bitstream ) {
  TRACE_BITSTREAM( "%s \n", __func__ );
  bitstream.write( (uint32_t)gfp.getGeometrySmoothingParamsPresentFlag(), 1 );  // u(1)
  bitstream.write( (uint32_t)gfp.getGeometryScaleParamsPresentFlag(), 1 );      // u(1)
  bitstream.write( (uint32_t)gfp.getGeometryOffsetParamsPresentFlag(), 1 );     // u(1)
  bitstream.write( (uint32_t)gfp.getGeometryRotationParamsPresentFlag(), 1 );   // u(1)
  bitstream.write( (uint32_t)gfp.getGeometryPointSizeInfoPresentFlag(), 1 );    // u(1)
  bitstream.write( (uint32_t)gfp.getGeometryPointShapeInfoPresentFlag(), 1 );   // u(1)

  if ( gfp.getGeometrySmoothingParamsPresentFlag() ) {
    bitstream.write( (uint32_t)gfp.getGeometrySmoothingEnabledFlag(), 1 );  // u(1)
    if ( gfp.getGeometrySmoothingEnabledFlag() ) {
      bitstream.write( (uint32_t)gfp.getGeometrySmoothingGridSize(), 8 );   // u(8)
      bitstream.write( (uint32_t)gfp.getGeometrySmoothingThreshold(), 8 );  // u(8)
    }
  }
  if ( gfp.getGeometryScaleParamsPresentFlag() ) {
    for ( size_t d = 0; d < 3; d++ ) {
      bitstream.write( (uint32_t)gfp.getGeometryScaleOnAxis( d ), 32 );  // u(32)
    }
  }
  if ( gfp.getGeometryOffsetParamsPresentFlag() ) {
    for ( size_t d = 0; d < 3; d++ ) {
      bitstream.writeS( gfp.getGeometryOffsetOnAxis( d ), 32 );  // i32
    }
  }
  if ( gfp.getGeometryRotationParamsPresentFlag() ) {
    for ( size_t d = 0; d < 4; d++ ) {
      bitstream.writeS( gfp.getGeometryRotationQuaternion( d ), 32 );  // i32
    }
  }
  if ( gfp.getGeometryPointSizeInfoPresentFlag() ) {
    bitstream.write( (uint32_t)gfp.getGeometryPointSizeInfo(), 16 );  // u(16)
  }
  if ( gfp.getGeometryPointShapeInfoPresentFlag() ) {
    bitstream.write( (uint32_t)gfp.getGeometryPointShapeInfo(), 4 );  // u(4)
  }
}

// 7.3.5.6 Patch frame attribute parameter set syntax
void PCCBitstreamEncoder::patchFrameAttributeParameterSet( PatchDataGroup&       pdg,
                                                           size_t                index,
                                                           SequenceParameterSet& sps,
                                                           PCCBitstream&         bitstream ) {
  TRACE_BITSTREAM( "%s \n", __func__ );
  auto& ai    = sps.getAttributeInformation();
  auto& pfaps = pdg.getPatchFrameAttributeParameterSet( index );
  auto& psps  = pdg.getPatchSequenceParameterSet( pfaps.getPatchSequencParameterSetId() );

  bitstream.writeUvlc( pfaps.getPatchFrameAttributeParameterSetId() );  // ue(v)
  bitstream.writeUvlc( pfaps.getPatchSequencParameterSetId() );         //  ue(v)

  TRACE_BITSTREAM( "PatchFrameAttributeParameterSetId = %u  \n", pfaps.getPatchFrameAttributeParameterSetId() );
  TRACE_BITSTREAM( "PatchSequencParameterSetId       = %u  \n", pfaps.getPatchSequencParameterSetId() );

  size_t attributeDimension = ai.getAttributeDimensionMinus1( pfaps.getPatchFrameAttributeParameterSetId() ) + 1;
  TRACE_BITSTREAM( " attributeDimension = %lu \n", attributeDimension );
  if ( ai.getAttributeParamsEnabledFlag() ) {
    attributeFrameParams( pfaps.getAttributeFrameParams(), attributeDimension, bitstream );
  }
  if ( ai.getAttributePatchParamsEnabledFlag() ) {
    bitstream.write( (uint32_t)pfaps.getAttributePatchScaleParamsEnabledFlag(), 1 );   //  u(1)
    bitstream.write( (uint32_t)pfaps.getAttributePatchOffsetParamsEnabledFlag(), 1 );  // u(1)
  }
  byteAlignment( bitstream );
}

// 7.3.5.7 Attribute frame Params syntax TODO: add attributeDimensions loop, remove afp.getAttributeSmoothingRadius,
// afp.getAttributeSmoothingNeighbourCount and afp.getAttributeSmoothingRadius2BoundaryDetection, add three missing
// parameters
void PCCBitstreamEncoder::attributeFrameParams( AttributeFrameParams& afp,
                                                size_t                attributeDimension,
                                                PCCBitstream&         bitstream ) {
  TRACE_BITSTREAM( "%s \n", __func__ );
  bitstream.write( (uint32_t)afp.getAttributeSmoothingParamsPresentFlag(), 1 );  // u(1)
  bitstream.write( (uint32_t)afp.getAttributeScaleParamsPresentFlag(), 1 );      // u(1)
  bitstream.write( (uint32_t)afp.getAttributeOffsetParamsPresentFlag(), 1 );     // u(1)
  if ( afp.getAttributeSmoothingParamsPresentFlag() ) {
    bitstream.write( (uint32_t)afp.getAttributeGridSmoothingEnabledFlag(), 1 );           // u(1)
    bitstream.write( (uint32_t)afp.getAttributeSmoothingRadius(), 8 );                    //  u(8)
    bitstream.write( (uint32_t)afp.getAttributeSmoothingNeighbourCount(), 8 );            //   u(8)
    bitstream.write( (uint32_t)afp.getAttributeSmoothingRadius2BoundaryDetection(), 8 );  //  u(8)
    bitstream.write( (uint32_t)afp.getAttributeSmoothingThreshold(), 8 );                 // u(8)
    if ( afp.getAttributeGridSmoothingEnabledFlag() ) {
      bitstream.write( (uint32_t)afp.getAttributeSmoothingThresholdColorDifference(), 8 );  // u(8)
      bitstream.write( (uint32_t)afp.getAttributeSmoothingThresholdColorVariation(), 8 );   // u(8)
      bitstream.write( (uint32_t)afp.getAttributeSmoothingThresholdLocalEntropy(), 3 );     // u(3)
      bitstream.write( (uint32_t)afp.getAttributeSmoothingGridSize(), 8 );
    }
  }
  if ( afp.getAttributeScaleParamsPresentFlag() ) {
    for ( size_t i = 0; i < attributeDimension; i++ ) {
      bitstream.write( (uint32_t)afp.getAttributeScale( i ), 32 );  // u(32)
    }
  }
  if ( afp.getAttributeOffsetParamsPresentFlag() ) {
    for ( size_t i = 0; i < attributeDimension; i++ ) bitstream.writeS( afp.getAttributeOffset( i ), 32 );  // i32
  }
}

// 7.3.5.8 Geometry patch parameter set syntax
void PCCBitstreamEncoder::geometryPatchParameterSet( PatchDataGroup& pdg, size_t index, PCCBitstream& bitstream ) {
  TRACE_BITSTREAM( "%s \n", __func__ );
  auto& gpps  = pdg.getGeometryPatchParameterSet( index );
  auto& pfgps = pdg.getPatchFrameGeometryParameterSet( gpps.getPatchFrameGeometryParameterSetId() );
  bitstream.writeUvlc( gpps.getGeometryPatchParameterSetId() );       // ue(v)
  bitstream.writeUvlc( gpps.getPatchFrameGeometryParameterSetId() );  //  ue(v)
  if ( pfgps.getGeometryPatchScaleParamsEnabledFlag() || pfgps.getGeometryPatchOffsetParamsEnabledFlag() ||
       pfgps.getGeometryPatchRotationParamsEnabledFlag() || pfgps.getGeometryPatchPointSizeInfoEnabledFlag() ||
       pfgps.getGeometryPatchPointShapeInfoEnabledFlag() ) {
    bitstream.write( (uint32_t)gpps.getGeometryPatchParamsPresentFlag(), 1 );  // u(1)
    if ( gpps.getGeometryPatchParamsPresentFlag() )
      geometryPatchParams( gpps.getGeometryPatchParams(), pfgps, bitstream );
  }
  byteAlignment( bitstream );
}

// 7.3.5.9 Geometry patch Params syntax TODO: rotation should be defined using quaternions (dimension 4)
void PCCBitstreamEncoder::geometryPatchParams( GeometryPatchParams&            gpp,
                                               PatchFrameGeometryParameterSet& gfps,
                                               PCCBitstream&                   bitstream ) {
  TRACE_BITSTREAM( "%s \n", __func__ );
  if ( gfps.getGeometryPatchScaleParamsEnabledFlag() ) {
    bitstream.write( (uint32_t)gpp.getGeometryPatchScaleParamsPresentFlag(), 1 );  // u(1)
    if ( gpp.getGeometryPatchScaleParamsPresentFlag() ) {
      for ( size_t d = 0; d < 3; d++ ) {
        bitstream.write( (uint32_t)gpp.getGeometryPatchScaleOnAxis( d ), 32 );  // u(32)
      }
    }
  }
  if ( gfps.getGeometryPatchOffsetParamsEnabledFlag() ) {
    bitstream.write( (uint32_t)gpp.getGeometryPatchOffsetParamsPresentFlag(), 1 );  // u(1)
    if ( gpp.getGeometryPatchOffsetParamsPresentFlag() ) {
      for ( size_t d = 0; d < 3; d++ ) {
        bitstream.writeS( gpp.getGeometryPatchOffsetOnAxis( d ), 32 );  // i(32)
      }
    }
  }
  if ( gfps.getGeometryPatchRotationParamsEnabledFlag() ) {
    bitstream.write( (uint32_t)gpp.getGeometryPatchRotationParamsPresentFlag(), 1 );  // u(1)
    if ( gpp.getGeometryPatchRotationParamsPresentFlag() ) {
      for ( size_t d = 0; d < 4; d++ ) {
        bitstream.writeS( gpp.getGeometryPatchRotationQuaternion( d ), 32 );  // i(32)
      }
    }
  }
  if ( gfps.getGeometryPatchPointSizeInfoEnabledFlag() ) {
    bitstream.write( (uint32_t)gpp.getGeometryPatchPointSizeInfoPresentFlag(), 1 );  // u(1)
    if ( gpp.getGeometryPatchPointSizeInfoPresentFlag() ) {
      bitstream.write( (uint32_t)gpp.getGeometryPatchPointSizeInfo(), 16 );  // u(16)
    }
  }
  if ( gfps.getGeometryPatchPointShapeInfoEnabledFlag() ) {
    bitstream.write( (uint32_t)gpp.getGeometryPatchPointShapeInfoPresentFlag(), 1 );  // u(1)
    if ( gpp.getGeometryPatchPointShapeInfoPresentFlag() ) {
      bitstream.write( (uint32_t)gpp.getGeometryPatchPointShapeInfo(), 4 );  // u(14
    }
  }
}

// 7.3.5.10 Attribute patch parameter set syntax TODO: add apps.setAttributeDimensionMinus1() u(8)
void PCCBitstreamEncoder::attributePatchParameterSet( PatchDataGroup&       pdg,
                                                      size_t                index,
                                                      SequenceParameterSet& sps,
                                                      PCCBitstream&         bitstream ) {
  TRACE_BITSTREAM( "%s \n", __func__ );
  auto& ai    = sps.getAttributeInformation();
  auto& apps  = pdg.getAttributePatchParameterSet( index );
  auto& pfaps = pdg.getPatchFrameAttributeParameterSet( apps.getPatchFrameAttributeParameterSetId() );
  bitstream.writeUvlc( apps.getAttributePatchParameterSetId() );
  bitstream.writeUvlc( apps.getPatchFrameAttributeParameterSetId() );
  size_t attributeDimension = ai.getAttributeDimensionMinus1( apps.getAttributePatchParameterSetId() ) + 1;
  if ( pfaps.getAttributePatchScaleParamsEnabledFlag() || pfaps.getAttributePatchOffsetParamsEnabledFlag() ) {
    bitstream.write( (uint32_t)apps.getAttributePatchParamsPresentFlag(), 1 );  // u(1)
    if ( apps.getAttributePatchParamsPresentFlag() ) {
      attributePatchParams( apps.getAttributePatchParams(), pfaps, attributeDimension, bitstream );
    }
  }
  byteAlignment( bitstream );
}

// 7.3.5.11 Attribute patch Params syntax (apps)
void PCCBitstreamEncoder::attributePatchParams( AttributePatchParams&            app,
                                                PatchFrameAttributeParameterSet& afps,
                                                size_t                           dimension,
                                                PCCBitstream&                    bitstream ) {
  TRACE_BITSTREAM( "%s \n", __func__ );
  if ( afps.getAttributePatchScaleParamsEnabledFlag() ) {
    bitstream.write( (uint32_t)app.getAttributePatchScaleParamsPresentFlag(), 1 );  //  u(1)
    if ( app.getAttributePatchScaleParamsPresentFlag() ) {
      for ( size_t i = 0; i < dimension; i++ ) {
        bitstream.write( (uint32_t)app.getAttributePatchScale( i ), 32 );  //  u(32)
      }
    }
  }
  if ( afps.getAttributePatchOffsetParamsEnabledFlag() ) {
    bitstream.write( (uint32_t)app.getAttributePatchOffsetParamsPresentFlag(), 1 );  //   u(1)
    if ( app.getAttributePatchOffsetParamsPresentFlag() ) {
      for ( size_t i = 0; i < dimension; i++ ) {
        bitstream.writeS( app.getAttributePatchOffset( i ), 32 );  // i(32)
      }
    }
  }
}

// 7.3.5.12 Patch frame parameter set syntax TODO: add pfps.getGeometryPatchFrameParameterSetId and
// pfps.getAttributePatchFrameParameterSetId[attributeCount], and remove pfps.getLocalOverrideGeometryPatchEnableFlag
// and pfps.getLocalOverrideAttributePatchEnableFlag
void PCCBitstreamEncoder::patchFrameParameterSet( PatchDataGroup&       pdg,
                                                  size_t                index,
                                                  SequenceParameterSet& sps,
                                                  PCCBitstream&         bitstream ) {
  TRACE_BITSTREAM( "%s \n", __func__ );
  auto& ai   = sps.getAttributeInformation();
  auto& pfps = pdg.getPatchFrameParameterSet( index );
  auto& psps = pdg.getPatchSequenceParameterSet( pfps.getPatchSequenceParameterSetId() );

  bitstream.writeUvlc( pfps.getPatchFrameParameterSetId() );             // ue(v)
  bitstream.writeUvlc( pfps.getPatchSequenceParameterSetId() );          // ue(v)
  bitstream.write( pfps.getLocalOverrideGeometryPatchEnableFlag(), 1 );  // u(1)
  for ( size_t i = 0; i < ai.getAttributeCount(); i++ ) {
    bitstream.write( pfps.getLocalOverrideAttributePatchEnableFlag( i ), 1 );  // u(1)
  }
  bitstream.writeUvlc( pfps.getAdditionalLtPfocLsbLen() );  // ue(v)
  if ( sps.getProjection45DegreeEnableFlag() ) {
    bitstream.write( pfps.getProjection45DegreeEnableFlag(), 1 );  // u(1)
  }

  byteAlignment( bitstream );
}

// 7.3.5.13 Patch frame layer unit syntax
void PCCBitstreamEncoder::patchTileGroupLayerUnit( PatchDataGroup& pdg,
                                                   size_t          index,
                                                   PCCContext&     context,
                                                   PCCBitstream&   bitstream ) {
  TRACE_BITSTREAM( "%s \n", __func__ );
  auto& ptglu    = pdg.getPatchTileGroupLayerUnit( index );
  auto& pfluPrev = pdg.getPatchTileGroupLayerUnit( ( std::max )( 0, (int32_t)index - 1 ) );
  ptglu.getPatchTileGroupHeader().setFrameIndex( ptglu.getFrameIndex() );
  ptglu.getPatchTileGroupDataUnit().setFrameIndex( ptglu.getFrameIndex() );
  patchTileGroupHeader( ptglu.getPatchTileGroupHeader(), pfluPrev.getPatchTileGroupHeader(), context, bitstream );
  patchTileGroupDataUnit( ptglu.getPatchTileGroupDataUnit(), ptglu.getPatchTileGroupHeader(), context, bitstream );
}

// 7.3.5.14 Patch frame header syntax TODO: difference between ue(v) and u(v)?
void PCCBitstreamEncoder::patchTileGroupHeader( PatchTileGroupHeader& ptgh,
                                                PatchTileGroupHeader& pfhPrev,
                                                PCCContext&           context,
                                                PCCBitstream&         bitstream ) {
  TRACE_BITSTREAM( "%s \n", __func__ );
  auto& pdg  = context.getPatchDataGroup();
  auto& psps = pdg.getPatchSequenceParameterSet( ptgh.getPatchFrameParameterSetId() );
  auto& pfps = pdg.getPatchFrameParameterSet( ptgh.getPatchFrameParameterSetId() );
  auto& gi   = context.getSps().getGeometryInformation();
  auto& sps  = context.getSps();

  bitstream.writeUvlc( ptgh.getPatchFrameParameterSetId() );  // ue(v )
  bitstream.writeUvlc( ptgh.getAddress() );                   // u( v )
  bitstream.writeUvlc( ptgh.getType() );                      // ue( v )
  bitstream.writeUvlc( ptgh.getPatchFrameOrderCntLsb() );     // u( v )

  TRACE_BITSTREAM( "Id     = %u \n", ptgh.getPatchFrameParameterSetId() );
  TRACE_BITSTREAM( "Adress = %u \n", ptgh.getAddress() );
  TRACE_BITSTREAM( "Type   = %u \n", ptgh.getType() );
  TRACE_BITSTREAM( "POC    = %u \n", ptgh.getPatchFrameOrderCntLsb() );
  TRACE_BITSTREAM( "psps.getNumRefPatchFrameListsInPsps() = %lu\n", psps.getNumRefPatchFrameListsInPsps() );
  TRACE_BITSTREAM( "psps.getNumRefPatchFrameListsInPsps() = %lu \n", psps.getNumRefPatchFrameListsInPsps() );
  TRACE_BITSTREAM( "gi.getGeometry3dCoordinatesBitdepthMinus1() = %lu \n",
                   gi.getGeometry3dCoordinatesBitdepthMinus1() );

  if ( psps.getNumRefPatchFrameListsInPsps() > 0 ) {
    bitstream.write( (uint32_t)ptgh.getRefPatchFrameListSpsFlag(), 1 );  // u( 1 )
  }
  if ( ptgh.getRefPatchFrameListSpsFlag() ) {
    if ( psps.getNumRefPatchFrameListsInPsps() > 1 ) {
      bitstream.writeUvlc( ptgh.getRefPatchFrameListIdx() );  // u( v )
    } else {
      psps.getRefListStruct( psps.getNumRefPatchFrameListsInPsps() );
    }
    uint8_t rlsIdx =
        psps.getNumRefPatchFrameListsInPsps() ? ptgh.getRefPatchFrameListIdx() : psps.getNumRefPatchFrameListsInPsps();
    size_t numLtrpEntries = 0;
    for ( size_t i = 0; i < psps.getRefListStruct( rlsIdx ).getNumRefEntries(); i++ ) {
      if ( !psps.getRefListStruct( rlsIdx ).getStRefPatchFrameFlag( i ) ) { numLtrpEntries++; }
    }

    for ( size_t j = 0; j < numLtrpEntries; j++ ) {
      bitstream.write( (uint32_t)ptgh.getAdditionalPfocLsbPresentFlag( j ), 1 );  // u( 1 )
      if ( ptgh.getAdditionalPfocLsbPresentFlag( j ) ) {
        bitstream.writeUvlc( ptgh.getAdditionalPfocLsbVal( j ) );  // u( v )
      }
    }

    if ( ptgh.getType() == PATCH_FRAME_P && psps.getRefListStruct( rlsIdx ).getNumRefEntries() > 1 ) {
      bitstream.write( (uint32_t)ptgh.getNumRefIdxActiveOverrideFlag(), 1 );  // u( 1 )
      if ( ptgh.getNumRefIdxActiveOverrideFlag() ) {
        bitstream.writeUvlc( ptgh.getNumRefIdxActiveMinus1() );  // u( v )
      }
    }
  }
  auto          geometryBitDepth2D = context.getSps().getGeometryInformation().getGeometryNominal2dBitdepthMinus1() + 1;
  const uint8_t maxBitCountForMaxDepth =
      uint8_t( geometryBitDepth2D - gbitCountSize[context.getSps().getMinLevel()] + 1 );  // 8

  if ( pfps.getProjection45DegreeEnableFlag() == 0 ) {
    ptgh.setInterPredictPatch2dDeltaSizeDBitCountMinus1( maxBitCountForMaxDepth );
  } else {
    ptgh.setInterPredictPatch2dDeltaSizeDBitCountMinus1( maxBitCountForMaxDepth + 1 );
  }

  if ( ptgh.getType() == PATCH_FRAME_I ) {
    bitstream.write( (uint32_t)ptgh.getInterPredictPatch2dShiftUBitCountMinus1(), 8 );              // u( 8 )
    bitstream.write( (uint32_t)ptgh.getInterPredictPatch2dShiftVBitCountMinus1(), 8 );              // u( 8 )
    bitstream.write( (uint32_t)ptgh.getInterPredictPatch3dShiftTangentAxisBitCountMinus1(), 8 );    // u( 8 )
    bitstream.write( (uint32_t)ptgh.getInterPredictPatch3dShiftBitangentAxisBitCountMinus1(), 8 );  // u( 8 )
    bitstream.write( (uint32_t)ptgh.getInterPredictPatch3dShiftNormalAxisBitCountMinus1(), 8 );     // u( 8 )
    bitstream.write( (uint32_t)ptgh.getInterPredictPatchLodBitCount(), 8 );                         // u( 8 )
  } else {
    bool countFlag[6] = {0, 0, 0, 0, 0, 0};
    bool countSumFlag = 0;
    countFlag[0] =
        ptgh.getInterPredictPatch2dShiftUBitCountMinus1() > pfhPrev.getInterPredictPatch2dShiftUBitCountMinus1();
    countFlag[1] =
        ptgh.getInterPredictPatch2dShiftVBitCountMinus1() > pfhPrev.getInterPredictPatch2dShiftVBitCountMinus1();
    countFlag[2] = ptgh.getInterPredictPatch3dShiftTangentAxisBitCountMinus1() >
                   pfhPrev.getInterPredictPatch3dShiftTangentAxisBitCountMinus1();
    countFlag[3] = ptgh.getInterPredictPatch3dShiftBitangentAxisBitCountMinus1() >
                   pfhPrev.getInterPredictPatch3dShiftBitangentAxisBitCountMinus1();
    countFlag[4] = ptgh.getInterPredictPatch3dShiftNormalAxisBitCountMinus1() >
                   pfhPrev.getInterPredictPatch3dShiftNormalAxisBitCountMinus1();
    countFlag[5] = ptgh.getInterPredictPatchLodBitCount() > pfhPrev.getInterPredictPatchLodBitCount();
    for ( size_t i = 0; i < 6; i++ ) countSumFlag |= countFlag[i];
    ptgh.setInterPredictPatchBitCountFlag( countSumFlag );
    ptgh.setInterPredictPatch2dShiftUBitCountFlag( countFlag[0] );
    ptgh.setInterPredictPatch2dShiftVBitCountFlag( countFlag[1] );
    ptgh.setInterPredictPatch3dShiftTangentAxisBitCountFlag( countFlag[2] );
    ptgh.setInterPredictPatch3dShiftBitangentAxisBitCountFlag( countFlag[3] );
    ptgh.setInterPredictPatch3dShiftNormalAxisBitCountFlag( countFlag[4] );
    ptgh.setInterPredictPatchLodBitCountFlag( countFlag[5] );
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
    if ( !ptgh.getInterPredictPatchBitCountFlag() || !ptgh.getInterPredictPatch3dShiftNormalAxisBitCountFlag() ) {
      ptgh.setInterPredictPatch3dShiftNormalAxisBitCountMinus1(
          pfhPrev.getInterPredictPatch3dShiftNormalAxisBitCountMinus1() );
    }
    if ( !ptgh.getInterPredictPatchBitCountFlag() || !ptgh.getInterPredictPatchLodBitCountFlag() ) {
      ptgh.setInterPredictPatchLodBitCount( pfhPrev.getInterPredictPatchLodBitCount() );
    }
    bitstream.write( (uint32_t)ptgh.getInterPredictPatchBitCountFlag(), 1 );  // u( 1 )
    if ( ptgh.getInterPredictPatchBitCountFlag() ) {
      bitstream.write( (uint32_t)ptgh.getInterPredictPatch2dShiftUBitCountFlag(), 1 );  // u( 1 )
      if ( ptgh.getInterPredictPatch2dShiftUBitCountFlag() ) {
        bitstream.write( (uint32_t)ptgh.getInterPredictPatch2dShiftUBitCountMinus1(), 8 );  // u( 8 )
      }
      bitstream.write( (uint32_t)ptgh.getInterPredictPatch2dShiftVBitCountFlag(), 1 );  // u( 1 )
      if ( ptgh.getInterPredictPatch2dShiftVBitCountFlag() ) {
        bitstream.write( (uint32_t)ptgh.getInterPredictPatch2dShiftVBitCountMinus1(), 8 );  // u( 8 )
      }
      bitstream.write( (uint32_t)ptgh.getInterPredictPatch3dShiftTangentAxisBitCountFlag(), 1 );  // u( 1 )
      if ( ptgh.getInterPredictPatch3dShiftTangentAxisBitCountFlag() ) {
        bitstream.write( (uint32_t)ptgh.getInterPredictPatch3dShiftTangentAxisBitCountMinus1(), 8 );  // u( 8 )
      }
      bitstream.write( (uint32_t)ptgh.getInterPredictPatch3dShiftBitangentAxisBitCountFlag(), 1 );  // u( 1 )
      if ( ptgh.getInterPredictPatch3dShiftBitangentAxisBitCountFlag() ) {
        bitstream.write( (uint32_t)ptgh.getInterPredictPatch3dShiftBitangentAxisBitCountMinus1(), 8 );  // u( 8 )
      }
      bitstream.write( (uint32_t)ptgh.getInterPredictPatch3dShiftNormalAxisBitCountFlag(), 1 );  // u( 1 )
      if ( ptgh.getInterPredictPatch3dShiftNormalAxisBitCountFlag() ) {
        bitstream.write( (uint32_t)ptgh.getInterPredictPatch3dShiftNormalAxisBitCountMinus1(), 8 );  // u( 8 )
      }
      bitstream.write( (uint32_t)ptgh.getInterPredictPatchLodBitCountFlag(), 1 );  // u( 1 )
      if ( ptgh.getInterPredictPatchLodBitCountFlag() ) {
        bitstream.write( (uint32_t)ptgh.getInterPredictPatchLodBitCount(), 8 );  // u( 8 )
      }
    }
  }
  if ( sps.getPcmPatchEnabledFlag() ) {
    bitstream.write( (uint32_t)ptgh.getPcm3dShiftBitCountPresentFlag(), 1 );  // u( 1 )
    if ( ptgh.getPcm3dShiftBitCountPresentFlag() ) {
      // bitstream.write((uint32_t)ptgh.getPcm3dShiftAxisBitCountMinus1(), 11);
      bitstream.write( (uint32_t)ptgh.getPcm3dShiftAxisBitCountMinus1(),
                       gi.getGeometry3dCoordinatesBitdepthMinus1() + 1 );
    }
  }
  TRACE_BITSTREAM(
      "InterPredictPatchBitCount Flag %d %d %d %d %d %d %d %d Count = %u %u %u %u %u %u %u \n",
      ptgh.getInterPredictPatchBitCountFlag(), ptgh.getInterPredictPatch2dShiftUBitCountFlag(),
      ptgh.getInterPredictPatch2dShiftVBitCountFlag(), ptgh.getInterPredictPatch3dShiftTangentAxisBitCountFlag(),
      ptgh.getInterPredictPatch3dShiftBitangentAxisBitCountFlag(),
      ptgh.getInterPredictPatch3dShiftNormalAxisBitCountFlag(), ptgh.getInterPredictPatchLodBitCountFlag(),
      ptgh.getPcm3dShiftBitCountPresentFlag(), ptgh.getInterPredictPatch2dShiftUBitCountMinus1(),
      ptgh.getInterPredictPatch2dShiftVBitCountMinus1(), ptgh.getInterPredictPatch3dShiftTangentAxisBitCountMinus1(),
      ptgh.getInterPredictPatch3dShiftBitangentAxisBitCountMinus1(),
      ptgh.getInterPredictPatch3dShiftNormalAxisBitCountMinus1(), ptgh.getInterPredictPatchLodBitCount(),
      ptgh.getPcm3dShiftAxisBitCountMinus1() );
  byteAlignment( bitstream );
}

// 7.3.5.15 Reference list structure syntax TODO: difference between ue(v) and u(v)?
void PCCBitstreamEncoder::refListStruct( RefListStruct&             rls,
                                         PatchSequenceParameterSet& psps,
                                         PCCBitstream&              bitstream ) {
  TRACE_BITSTREAM( "%s \n", __func__ );
  bitstream.writeUvlc( rls.getNumRefEntries() );  // ue(v)
  rls.allocate();
  for ( size_t i = 0; i < rls.getNumRefEntries(); i++ ) {
    if ( psps.getLongTermRefPatchFramesFlag() ) {
      bitstream.write( rls.getStRefPatchFrameFlag( i ), 1 );  // u(1)
      if ( rls.getStRefPatchFrameFlag( i ) ) {
        bitstream.writeUvlc( rls.getAbsDeltaPfocSt( i ) );  // ue(v)
        if ( rls.getAbsDeltaPfocSt( i ) > 0 ) {
          bitstream.write( rls.getStrpfEntrySignFlag( i ), 1 );  // u(1)
        } else {
          bitstream.writeUvlc( rls.getPfocLsbLt( i ) );  // u(v)
        }
      }
    }
  }
}

// 7.3.5.16 Patch frame data unit syntax TODO: modify loop to use patchMode instead of flag
void PCCBitstreamEncoder::patchTileGroupDataUnit( PatchTileGroupDataUnit& ptgdu,
                                                  PatchTileGroupHeader&   ptgh,
                                                  PCCContext&             context,
                                                  PCCBitstream&           bitstream ) {
  TRACE_BITSTREAM( "%s \n", __func__ );
  TRACE_BITSTREAM( "ptgh.getType()        = %lu \n", ptgh.getType() );
#if !LAST_PATCH_HLS
  uint8_t       moreAvailablePatchFlag = ptgdu.getPatchCount() > 0;
  const uint8_t bitCountPatchMode      = ( PCCPatchFrameType( ptgh.getType() ) ) == PATCH_FRAME_I ? 1 : 2;
  bitstream.write( moreAvailablePatchFlag, 1 );
  TRACE_BITSTREAM( "moreAvailablePatchFlag = %d \n", moreAvailablePatchFlag );
  TRACE_BITSTREAM( "bitCountPatchMode = %u \n", bitCountPatchMode );
#endif
  for ( size_t puCount = 0; puCount < ptgdu.getPatchCount(); puCount++ ) {
#if LAST_PATCH_HLS
    bitstream.writeSvlc( uint32_t( ptgdu.getPatchMode( puCount ) ) );
#else
    bitstream.write( uint32_t( ptgdu.getPatchMode( puCount ) ), bitCountPatchMode );
#endif
    TRACE_BITSTREAM( "patchMode = %lu \n", ptgdu.getPatchMode( puCount ) );
    ptgdu.getPatchInformationData( puCount ).setFrameIndex( ptgdu.getFrameIndex() );
    ptgdu.getPatchInformationData( puCount ).setPatchIndex( puCount );
    patchInformationData( ptgdu.getPatchInformationData( puCount ), ptgdu.getPatchMode( puCount ), ptgh, context,
                          bitstream );
#if !LAST_PATCH_HLS
    moreAvailablePatchFlag = !( ( puCount + 1 ) == ptgdu.getPatchCount() );
    bitstream.write( moreAvailablePatchFlag, 1 );  // ae(v)
    TRACE_BITSTREAM( "moreAvailablePatchFlag = %d \n", moreAvailablePatchFlag );
#endif
  }
  byteAlignment( bitstream );
}

// 7.3.5.17 Patch information data syntax TODO: gppsId and appsId using u(v) instead of ue(v) ?
void PCCBitstreamEncoder::patchInformationData( PatchInformationData& pid,
                                                size_t                patchMode,
                                                PatchTileGroupHeader& ptgh,
                                                PCCContext&           context,
                                                PCCBitstream&         bitstream ) {
  TRACE_BITSTREAM( "%s \n", __func__ );
  auto&         sps             = context.getSps();
  auto&         ai              = sps.getAttributeInformation();
  auto&         pdg             = context.getPatchDataGroup();
  auto&         pfps            = pdg.getPatchFrameParameterSet( ptgh.getPatchFrameParameterSetId() );
  const uint8_t bitCountGAppsId = 6;
  if ( ( PCCPatchFrameType( ptgh.getType() ) ) == PATCH_FRAME_P && patchMode == PATCH_MODE_P_SKIP ) {
    // skip mode.
    // currently not supported but added it for convenience. Could easily be removed
  } else if ( ( ( PCCPatchFrameType( ptgh.getType() ) ) == PATCH_FRAME_I && patchMode == PATCH_MODE_I_INTRA ) ||
              ( ( PCCPatchFrameType( ptgh.getType() ) ) == PATCH_FRAME_P && patchMode == PATCH_MODE_P_INTRA ) ) {
    if ( pfps.getLocalOverrideGeometryPatchEnableFlag() ) {
      bitstream.write( pid.getOverrideGeometryPatchFlag(), 1 );  // u(1)
      if ( pid.getOverrideGeometryPatchFlag() ) {
        bitstream.write( uint32_t( pid.getGeometryPatchParameterSetId() ), bitCountGAppsId );  // ue(v)
        TRACE_BITSTREAM( " gppsId = %lu \n", pid.getGeometryPatchParameterSetId() );
      }
    }
    TRACE_BITSTREAM( " ai.getAttributeCount() = %lu \n", ai.getAttributeCount() );
    for ( int i = 0; i < ai.getAttributeCount(); i++ ) {
      TRACE_BITSTREAM( " overight flag = %lu \n", pfps.getLocalOverrideAttributePatchEnableFlag( i ) );
      if ( pfps.getLocalOverrideAttributePatchEnableFlag( i ) ) {
        bitstream.write( pid.getOverrideAttributePatchFlag( i ), 1 );
        TRACE_BITSTREAM( " overrideAttributePatchFlag = %lu \n", pid.getOverrideAttributePatchFlag( i ) );
      }
      TRACE_BITSTREAM( " overight patch flag = %lu \n", pid.getOverrideAttributePatchFlag( i ) );
      if ( pid.getOverrideAttributePatchFlag( i ) ) {
        bitstream.write( uint32_t( pid.getAttributePatchParameterSetId( i ) ), bitCountGAppsId );
        TRACE_BITSTREAM( " AttributePatchParameterSetId = %lu \n", pid.getAttributePatchParameterSetId( i ) );
      }
    }
    auto& pdu = pid.getPatchDataUnit();
    pdu.setPduFrameIndex( pid.getFrameIndex() );
    pdu.setPduPatchIndex( pid.getPatchIndex() );
    patchDataUnit( pdu, ptgh, context, bitstream );
  } else if ( ( PCCPatchFrameType( ptgh.getType() ) ) == PATCH_FRAME_P && patchMode == PATCH_MODE_P_INTER ) {
    auto& dpdu = pid.getDeltaPatchDataUnit();
    dpdu.setDpduFrameIndex( pid.getFrameIndex() );
    dpdu.setDpduPatchIndex( pid.getPatchIndex() );
    deltaPatchDataUnit( dpdu, ptgh, context, bitstream );
  } else if ( ( ( PCCPatchFrameType( ptgh.getType() ) ) == PATCH_FRAME_I && patchMode == PATCH_MODE_I_PCM ) ||
              ( ( PCCPatchFrameType( ptgh.getType() ) ) == PATCH_FRAME_P && patchMode == PATCH_MODE_P_PCM ) ) {
    auto& ppdu = pid.getPCMPatchDataUnit();
    ppdu.setPpduFrameIndex( pid.getFrameIndex() );
    ppdu.setPpduPatchIndex( pid.getPatchIndex() );
    pcmPatchDataUnit( ppdu, ptgh, context, bitstream );
  }
}

// 7.3.5.18 Patch data unit syntax
void PCCBitstreamEncoder::patchDataUnit( PatchDataUnit&        pdu,
                                         PatchTileGroupHeader& ptgh,
                                         PCCContext&           context,
                                         PCCBitstream&         bitstream ) {
  auto& sps = context.getSps();
  TRACE_BITSTREAM( "%s \n", __func__ );
  bitstream.write( uint32_t( pdu.get2DShiftU() ), ptgh.getInterPredictPatch2dShiftUBitCountMinus1() + 1 );
  bitstream.write( uint32_t( pdu.get2DShiftV() ), ptgh.getInterPredictPatch2dShiftVBitCountMinus1() + 1 );
  bitstream.write( uint32_t( pdu.get2DDeltaSizeD() ), ptgh.getInterPredictPatch2dDeltaSizeDBitCountMinus1() + 1 );
  bitstream.writeSvlc( int32_t( pdu.get2DDeltaSizeU() ) );  // The way it is implemented in TM
  bitstream.writeSvlc( int32_t( pdu.get2DDeltaSizeV() ) );  // The way it is implemented in TM
  bitstream.write( uint32_t( pdu.get3DShiftTangentAxis() ),
                   ptgh.getInterPredictPatch3dShiftTangentAxisBitCountMinus1() + 1 );
  bitstream.write( uint32_t( pdu.get3DShiftBiTangentAxis() ),
                   ptgh.getInterPredictPatch3dShiftBitangentAxisBitCountMinus1() + 1 );
  bitstream.write( uint32_t( pdu.get3DShiftNormalAxis() ),
                   ptgh.getInterPredictPatch3dShiftNormalAxisBitCountMinus1() + 1 );

  bitstream.write( uint32_t( pdu.getProjectPlane() ), 3 );  // 0,1,2(near 0,1,2) 3,4,5(far 0,1,2)
  auto& pdg  = context.getPatchDataGroup();
  auto& psps = pdg.getPatchSequenceParameterSet( 0 );
  if ( psps.getUseEightOrientationsFlag() ) {
    bitstream.write( pdu.getOrientationIndex(), 3 );
  } else {
    bitstream.write( pdu.getOrientationIndex(), 1 );
  }
  if ( ptgh.getInterPredictPatchLodBitCount() > 0 ) {
    bitstream.write( uint32_t( pdu.getLod() ), ptgh.getInterPredictPatchLodBitCount() );
  }
  if ( sps.getPointLocalReconstructionEnabledFlag() ) {
    pointLocalReconstructionData( pdu.getPointLocalReconstructionData(), context, bitstream );
  }
  auto& pfps = pdg.getPatchFrameParameterSet( 0 );
  if ( pfps.getProjection45DegreeEnableFlag() ) {
    bitstream.write( uint32_t( pdu.get45DegreeProjectionPresentFlag() ), 1 );
  }
  if ( pdu.get45DegreeProjectionPresentFlag() ) {
    bitstream.write( uint32_t( pdu.get45DegreeProjectionRotationAxis() ), 2 );
  }

  TRACE_BITSTREAM( "Patch => UV %4lu %4lu S=%4ld %4ld %4ld P=%lu O=%d A=%lu %lu %lu P45= %d %d \n ", pdu.get2DShiftU(),
                   pdu.get2DShiftV(), pdu.get2DDeltaSizeU(), pdu.get2DDeltaSizeV(), pdu.get2DDeltaSizeD(),
                   pdu.getProjectPlane(), pdu.getOrientationIndex(), pdu.get3DShiftTangentAxis(),
                   pdu.get3DShiftBiTangentAxis(), pdu.get3DShiftNormalAxis(), pdu.get45DegreeProjectionPresentFlag(),
                   pdu.get45DegreeProjectionRotationAxis() );
}

// 7.3.5.19  Delta Patch data unit syntax TODO: Missing 10-projection syntax element?
void PCCBitstreamEncoder::deltaPatchDataUnit( DeltaPatchDataUnit&   dpdu,
                                              PatchTileGroupHeader& ptgh,
                                              PCCContext&           context,
                                              PCCBitstream&         bitstream ) {
  auto& sps = context.getSps();
  TRACE_BITSTREAM( "%s \n", __func__ );
  bitstream.writeSvlc( int32_t( dpdu.getDeltaPatchIdx() ) );
  bitstream.writeSvlc( int32_t( dpdu.get2DDeltaShiftU() ) );
  bitstream.writeSvlc( int32_t( dpdu.get2DDeltaShiftV() ) );
  bitstream.writeSvlc( int32_t( dpdu.get2DDeltaSizeU() ) );
  bitstream.writeSvlc( int32_t( dpdu.get2DDeltaSizeV() ) );
  bitstream.writeSvlc( int32_t( dpdu.get2DDeltaSizeD() ) );
  bitstream.writeSvlc( int32_t( dpdu.get3DDeltaShiftTangentAxis() ) );
  bitstream.writeSvlc( int32_t( dpdu.get3DDeltaShiftBiTangentAxis() ) );
  bitstream.writeSvlc( int32_t( dpdu.get3DDeltaShiftNormalAxis() ) );
  if ( sps.getPointLocalReconstructionEnabledFlag() ) {
    pointLocalReconstructionData( dpdu.getPointLocalReconstructionData(), context, bitstream );
  }

  TRACE_BITSTREAM(
      "%zu frame %zu DeltaPatch => DeltaIdx = %d ShiftUV = %ld %ld DeltaSize = %ld %ld %ld Axis = %ld %ld %ld\n",
      dpdu.getDpduFrameIndex(), dpdu.getDpduPatchIndex(), dpdu.getDeltaPatchIdx(), dpdu.get2DDeltaShiftU(),
      dpdu.get2DDeltaShiftV(), dpdu.get2DDeltaSizeU(), dpdu.get2DDeltaSizeV(), dpdu.get2DDeltaSizeD(),
      dpdu.get3DDeltaShiftTangentAxis(), dpdu.get3DDeltaShiftBiTangentAxis(), dpdu.get3DDeltaShiftNormalAxis() );
}

// 7.3.5.20 PCM patch data unit syntax TODO: getPcmPoints is u(v) in CD, but is currently se(v)
void PCCBitstreamEncoder::pcmPatchDataUnit( PCMPatchDataUnit&     ppdu,
                                            PatchTileGroupHeader& ptgh,
                                            PCCContext&           context,
                                            PCCBitstream&         bitstream ) {
  TRACE_BITSTREAM( "%s \n", __func__ );

  auto& sps = context.getSps();
  auto& gi  = context.getSps().getGeometryInformation();
  if ( sps.getPcmSeparateVideoPresentFlag() ) { bitstream.write( ppdu.getPatchInPcmVideoFlag(), 1 ); }
  bitstream.write( uint32_t( ppdu.get2DShiftU() ), ptgh.getInterPredictPatch2dShiftUBitCountMinus1() + 1 );
  bitstream.write( uint32_t( ppdu.get2DShiftV() ), ptgh.getInterPredictPatch2dShiftVBitCountMinus1() + 1 );
  bitstream.writeSvlc( int32_t( ppdu.get2DDeltaSizeU() ) );
  bitstream.writeSvlc( int32_t( ppdu.get2DDeltaSizeV() ) );

  if ( ptgh.getPcm3dShiftBitCountPresentFlag() ) {
    bitstream.write( uint32_t( ppdu.get3DShiftTangentAxis() ), ptgh.getPcm3dShiftAxisBitCountMinus1() + 1 );
    bitstream.write( uint32_t( ppdu.get3DShiftBiTangentAxis() ), ptgh.getPcm3dShiftAxisBitCountMinus1() + 1 );
    bitstream.write( uint32_t( ppdu.get3DShiftNormalAxis() ), ptgh.getPcm3dShiftAxisBitCountMinus1() + 1 );
  } else {
    size_t bitCountPcmU1V1D1 = gi.getGeometry3dCoordinatesBitdepthMinus1() - gi.getGeometryNominal2dBitdepthMinus1();
    bitstream.write( uint32_t( ppdu.get3DShiftTangentAxis() ), bitCountPcmU1V1D1 );
    bitstream.write( uint32_t( ppdu.get3DShiftBiTangentAxis() ), bitCountPcmU1V1D1 );
    bitstream.write( uint32_t( ppdu.get3DShiftNormalAxis() ), bitCountPcmU1V1D1 );
  }

  bitstream.writeSvlc( int32_t( ppdu.getPcmPoints() ) );
  TRACE_BITSTREAM(
      "PCM Patch => UV %4lu %4lu  S=%4ld %4ld  UVD1=%4ld %4ld %4ld NumPcmPoints=%lu PatchInPcmVideoFlag=%d \n",
      ppdu.get2DShiftU(), ppdu.get2DShiftV(), ppdu.get2DDeltaSizeU(), ppdu.get2DDeltaSizeV(),
      ppdu.get3DShiftBiTangentAxis(), ppdu.get3DShiftBiTangentAxis(), ppdu.get3DShiftNormalAxis(), ppdu.getPcmPoints(),
      ppdu.getPatchInPcmVideoFlag() );
}

// 7.3.5.21 Point local reconstruction syntax
void PCCBitstreamEncoder::pointLocalReconstructionInformation( PointLocalReconstructionInformation& plri,
                                                               PCCContext&                          context,
                                                               PCCBitstream&                        bitstream ) {
  TRACE_BITSTREAM( "%s \n", __func__ );
  bitstream.write( uint32_t( plri.getPlrlNumberOfModesMinus1() ), 4 );
  TRACE_BITSTREAM( "  NumberOfModesMinus1 = %u \n", plri.getPlrlNumberOfModesMinus1() );
  for ( size_t i = 0; i <= plri.getPlrlNumberOfModesMinus1(); i++ ) {
    bitstream.write( uint32_t( plri.getPlrlInterpolateFlag( i ) ), 1 );
    bitstream.write( uint32_t( plri.getPlrlFillingFlag( i ) ), 1 );
    bitstream.write( uint32_t( plri.getPlrlMinimumDepth( i ) ), 2 );
    bitstream.write( uint32_t( plri.getPlrlNeighbourMinus1( i ) ), 2 );
    TRACE_BITSTREAM( "  Mode[%lu] = I = %d F = %d D = %u N = %u \n", i, plri.getPlrlInterpolateFlag( i ),
                     plri.getPlrlFillingFlag( i ), plri.getPlrlMinimumDepth( i ), plri.getPlrlNeighbourMinus1( i ) );
  }
  bitstream.writeUvlc( uint32_t( plri.getPlrBlockThresholdPerPatchMinus1() ) );
  TRACE_BITSTREAM( "  BlockThresholdPerPatchMinus1 = %u \n", plri.getPlrBlockThresholdPerPatchMinus1() );
}

void PCCBitstreamEncoder::pointLocalReconstructionData( PointLocalReconstructionData& plrd,
                                                        PCCContext&                   context,
                                                        PCCBitstream&                 bitstream ) {
  auto& plri = context.getSps().getPointLocalReconstructionInformation();
  TRACE_BITSTREAM( "%s \n", __func__ );
  TRACE_BITSTREAM( "WxH= %lu x %lu \n", plrd.getPlrBlockToPatchMapWidth(), plrd.getPlrBlockToPatchMapHeight() );

  const size_t  blockCount   = plrd.getPlrBlockToPatchMapWidth() * plrd.getPlrBlockToPatchMapHeight();
  const uint8_t bitCountMode = uint8_t( getFixedLengthCodeBitsCount( uint32_t( plri.getPlrlNumberOfModesMinus1() ) ) );
  TRACE_BITSTREAM( "  bitCountMode = %u \n", bitCountMode );

  if ( blockCount > plri.getPlrBlockThresholdPerPatchMinus1() + 1 ) {
    bitstream.write( uint32_t( plrd.getPlrLevelFlag() ), 1 );
  }
  TRACE_BITSTREAM( "  LevelFlag = %u \n", plrd.getPlrLevelFlag() );
  if ( plrd.getPlrLevelFlag() ) {
    bitstream.write( uint32_t( plrd.getPlrPresentFlag() ), 1 );
    if ( plrd.getPlrPresentFlag() ) { bitstream.write( uint32_t( plrd.getPlrModeMinus1() ), bitCountMode ); }
    TRACE_BITSTREAM( "  ModePatch: Present = %d ModeMinus1 = %d \n", plrd.getPlrPresentFlag(),
                     plrd.getPlrPresentFlag() ? (int32_t)plrd.getPlrModeMinus1() : -1 );
  } else {
    for ( size_t i = 0; i < blockCount; i++ ) {
      bitstream.write( uint32_t( plrd.getPlrBlockPresentFlag( i ) ), 1 );
      if ( plrd.getPlrBlockPresentFlag( i ) ) {
        bitstream.write( uint32_t( plrd.getPlrBlockModeMinus1( i ) ), bitCountMode );
      }
      TRACE_BITSTREAM( "  Mode[ %4lu / %4lu ]: Present = %d ModeMinus1 = %d \n", i, blockCount,
                       plrd.getPlrBlockPresentFlag( i ),
                       plrd.getPlrBlockPresentFlag( i ) ? plrd.getPlrBlockModeMinus1( i ) : -1 );
    }
  }
#ifdef BITSTREAM_TRACE
  for ( size_t v0 = 0; v0 < plrd.getPlrBlockToPatchMapHeight(); ++v0 ) {
    for ( size_t u0 = 0; u0 < plrd.getPlrBlockToPatchMapWidth(); ++u0 ) {
      size_t i = v0 * plrd.getPlrBlockToPatchMapWidth() + u0;
      TRACE_BITSTREAM( "Patch Block[ %2lu %2lu <=> %4lu ] / [ %2lu %2lu ] Level = %d Present = %d Mode = %d \n", u0, v0,
                       i, plrd.getPlrBlockToPatchMapWidth(), plrd.getPlrBlockToPatchMapHeight(), plrd.getPlrLevelFlag(),
                       plrd.getPlrLevelFlag() ? plrd.getPlrPresentFlag() : plrd.getPlrBlockPresentFlag( i ),
                       plrd.getPlrLevelFlag()
                           ? plrd.getPlrPresentFlag() ? (int32_t)plrd.getPlrModeMinus1() : -1
                           : plrd.getPlrBlockPresentFlag( i ) ? plrd.getPlrBlockModeMinus1( i ) : -1 );
    }
  }
#endif
}
