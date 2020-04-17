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
#include "PCCHighLevelSyntax.h"
#include "PCCBitstream.h"
#include "PCCVideoBitstream.h"
#include "PCCContext.h"
#include "PCCFrameContext.h"
#include "PCCPatch.h"
#include "PCCVideoDecoder.h"
#include "PCCGroupOfFrames.h"
#include <tbb/tbb.h>
#include "PCCDecoder.h"

using namespace pcc;
using namespace std;

PCCDecoder::PCCDecoder() {
#ifdef ENABLE_PAPI_PROFILING
  initPapiProfiler();
#endif
}
PCCDecoder::~PCCDecoder() = default;
void PCCDecoder::setParameters( const PCCDecoderParameters& params ) { params_ = params; }

int  PCCDecoder::decode( PCCContext&       context,
                         PCCGroupOfFrames& reconstructs,
                         int32_t           atlasIndex = 0 ) {
  if ( params_.nbThread_ > 0 ) { tbb::task_scheduler_init init( static_cast<int>( params_.nbThread_ ) ); }
#ifdef CODEC_TRACE
  setTrace( true );
  openTrace( stringFormat( "%s_GOF%u_patch_decode.txt", removeFileExtension( params_.compressedStreamPath_ ).c_str(),
                           context.getVps().getVpccParameterSetId() ) );
#endif
  createPatchFrameDataStructure( context );
#ifdef CODEC_TRACE
  closeTrace();
#endif

  PCCVideoDecoder   videoDecoder;
  std::stringstream path;
  auto&             sps  = context.getVps();
  auto&             ai   = sps.getAttributeInformation( atlasIndex );
  auto&             oi   = sps.getOccupancyInformation( atlasIndex );
  auto&             gi   = sps.getGeometryInformation( atlasIndex );
  auto&             asps = context.getAtlasSequenceParameterSet( 0 );
  path << removeFileExtension( params_.compressedStreamPath_ ) << "_dec_GOF" << sps.getVpccParameterSetId() << "_";
#ifdef CODEC_TRACE
  setTrace( true );
  openTrace( stringFormat( "%s_GOF%u_codec_decode.txt", removeFileExtension( params_.compressedStreamPath_ ).c_str(),
                           sps.getVpccParameterSetId() ) );
#endif
  auto&        plt               = sps.getProfileTierLevel();
  bool         use444CodecIo     = plt.getProfileCodecGroupIdc() == CODEC_GROUP_HEVC444;
  const size_t mapCount          = sps.getMapCountMinus1( atlasIndex ) + 1;
  auto&        videoBitstreamOM  = context.getVideoBitstream( VIDEO_OCCUPANCY );
  int          decodedBitDepthOM = 8;
  videoDecoder.decompress( context.getVideoOccupancyMap(), path.str(), context.size(), videoBitstreamOM,
                           params_.videoDecoderOccupancyMapPath_, context, decodedBitDepthOM,
                           params_.keepIntermediateFiles_, use444CodecIo, false, "", "" );
  // converting the decoded bitdepth to the nominal bitdepth
  context.getVideoOccupancyMap().convertBitdepth( decodedBitDepthOM, oi.getOccupancyNominal2DBitdepthMinus1() + 1,
                                                  oi.getOccupancyMSBAlignFlag() );

  if ( sps.getMultipleMapStreamsPresentFlag( atlasIndex ) ) {
    context.getVideoGeometryMultiple().resize( sps.getMapCountMinus1( atlasIndex ) + 1 );
    size_t totalGeoSize = 0;
    for ( int mapIndex = 0; mapIndex < sps.getMapCountMinus1( atlasIndex ) + 1; mapIndex++ ) {
      // Decompress D[mapIndex]
      int decodedBitDepth = gi.getGeometryNominal2dBitdepthMinus1() + 1;  // this should be extracted from the bitstream
      auto  geometryIndex = static_cast<PCCVideoType>( VIDEO_GEOMETRY_D0 + mapIndex );
      auto& videoBitstream = context.getVideoBitstream( geometryIndex );
      videoDecoder.decompress( context.getVideoGeometryMultiple()[mapIndex], path.str(), context.size(), videoBitstream,
                               params_.videoDecoderPath_, context, decodedBitDepth, params_.keepIntermediateFiles_,
                               use444CodecIo );
      context.getVideoGeometryMultiple()[mapIndex].convertBitdepth(
          decodedBitDepth, gi.getGeometryNominal2dBitdepthMinus1() + 1, gi.getGeometryMSBAlignFlag() );
      std::cout << "geometry D" << mapIndex << " video ->" << videoBitstream.size() << " B" << std::endl;
      totalGeoSize += videoBitstream.size();
    }
    std::cout << "total geometry video ->" << totalGeoSize << " B" << std::endl;
  } else {
    int   decodedBitDepthGeo = gi.getGeometryNominal2dBitdepthMinus1() + 1;
    auto& videoBitstream     = context.getVideoBitstream( VIDEO_GEOMETRY );
    videoDecoder.decompress( context.getVideoGeometry(), path.str(), context.size() * mapCount, videoBitstream,
                             params_.videoDecoderPath_, context, decodedBitDepthGeo, params_.keepIntermediateFiles_,
                             use444CodecIo );
    context.getVideoGeometry().convertBitdepth( decodedBitDepthGeo, gi.getGeometryNominal2dBitdepthMinus1() + 1,
                                                gi.getGeometryMSBAlignFlag() );
    std::cout << "geometry video ->" << videoBitstream.size() << " B" << std::endl;
  }

  if ( sps.getRawPatchEnabledFlag( atlasIndex ) && sps.getRawSeparateVideoPresentFlag( atlasIndex ) ) {
    int   decodedBitDepthMP = gi.getGeometryNominal2dBitdepthMinus1() + 1;
    auto& videoBitstreamMP  = context.getVideoBitstream( VIDEO_GEOMETRY_RAW );
    videoDecoder.decompress( context.getVideoRawPointsGeometry(), path.str(), context.size(), videoBitstreamMP,
                             params_.videoDecoderPath_, context, decodedBitDepthMP, params_.keepIntermediateFiles_ );
    context.getVideoRawPointsGeometry().convertBitdepth( decodedBitDepthMP, gi.getGeometryNominal2dBitdepthMinus1() + 1,
                                                   gi.getGeometryMSBAlignFlag() );
    // generateRawPointsGeometryfromVideo( context, reconstructs );
    std::cout << " raw points geometry -> " << videoBitstreamMP.size() << " B " << endl;
  }

  if ( ai.getAttributeCount() > 0 ) {
    for ( int attrIndex = 0; attrIndex < sps.getAttributeInformation( atlasIndex ).getAttributeCount();
          attrIndex++ ) {  // right now we only have one attribute, this should be generalized
      int decodedBitdepthAttribute   = ai.getAttributeNominal2dBitdepthMinus1( attrIndex ) + 1;
      int decodedBitdepthAttributeMP = ai.getAttributeNominal2dBitdepthMinus1( attrIndex ) + 1;
      printf("RawPatchEnabledFlag[ atlasIndex = %d ] = %d \n",atlasIndex,sps.getRawPatchEnabledFlag( atlasIndex ) );
      printf("ProfileCodecGroupIdc() = %d \n",(int)plt.getProfileCodecGroupIdc() );
      for ( int attrPartitionIndex = 0;
            attrPartitionIndex <
            sps.getAttributeInformation( atlasIndex ).getAttributeDimensionPartitionsMinus1( attrIndex ) + 1;
            attrPartitionIndex++ ) {  // right now we have only one partition, this should be generalized
        if ( sps.getMultipleMapStreamsPresentFlag( atlasIndex ) ) {
          int sizeTextureVideo = 0;
          context.getVideoTextureMultiple().resize(
              sps.getMapCountMinus1( atlasIndex ) +
              1 );  // this allocation is considering only one attribute, with a single partition, but multiple streams
          for ( int mapIndex = 0; mapIndex < sps.getMapCountMinus1( atlasIndex ) + 1; mapIndex++ ) {
            // decompress T[mapIndex]
            auto textureIndex =
                static_cast<PCCVideoType>( VIDEO_TEXTURE_T0 + attrPartitionIndex + MAX_NUM_ATTR_PARTITIONS * mapIndex );
            auto& videoBitstream = context.getVideoBitstream( textureIndex );
            videoDecoder.decompress( context.getVideoTextureMultiple()[mapIndex], path.str(), context.size(),
                                     videoBitstream, params_.videoDecoderPath_, context,
                                     ai.getAttributeNominal2dBitdepthMinus1( 0 ) + 1, params_.keepIntermediateFiles_,
                                     sps.getRawPatchEnabledFlag( atlasIndex ), params_.patchColorSubsampling_,
                                     params_.inverseColorSpaceConversionConfig_, params_.colorSpaceConversionPath_ );
            /*context.getVideoTextureMultiple()[mapIndex].convertBitdepth(
              decodedBitdepthAttribute, ai.getAttributeNominal2dBitdepthMinus1(attrIndex) + 1,
              ai.getAttributeMSBAlignFlag(attrIndex));*/
            std::cout << "texture T" << mapIndex << " video ->" << videoBitstream.size() << " B" << std::endl;
            sizeTextureVideo += videoBitstream.size();
          }
          std::cout << "texture    video ->" << sizeTextureVideo << " B" << std::endl;
        } else {
          auto  textureIndex   = static_cast<PCCVideoType>( VIDEO_TEXTURE + attrPartitionIndex );
          auto& videoBitstream = context.getVideoBitstream( textureIndex );
          videoDecoder.decompress( context.getVideoTexture(),                 // video,
                                   path.str(),                                // path,
                                   context.size() * mapCount,                 // frameCount,
                                   videoBitstream,                            // bitstream,
                                   params_.videoDecoderPath_,                 // decoderPath,
                                   context,                                   // contexts,
                                   decodedBitdepthAttribute,                  // bitDepth,
                                   params_.keepIntermediateFiles_,            // keepIntermediateFiles
                                   sps.getRawPatchEnabledFlag( atlasIndex ), // && plt.getProfileCodecGroupIdc() == CODEC_GROUP_HEVC444,  // use444CodecIo
                                   params_.patchColorSubsampling_,            // patchColorSubsampling
                                   params_.inverseColorSpaceConversionConfig_, params_.colorSpaceConversionPath_ );
          /*context.getVideoTexture().convertBitdepth(
              decodedBitdepthAttribute, ai.getAttributeNominal2dBitdepthMinus1(attrIndex) + 1,
             ai.getAttributeMSBAlignFlag(attrIndex));*/
          std::cout << "texture video  ->" << videoBitstream.size() << " B" << std::endl;
        }
        if ( sps.getRawPatchEnabledFlag( atlasIndex ) && sps.getRawSeparateVideoPresentFlag( atlasIndex ) ) {
          auto  textureIndex     = static_cast<PCCVideoType>( VIDEO_TEXTURE_RAW + attrPartitionIndex );
          auto& videoBitstreamMP = context.getVideoBitstream( textureIndex );
          videoDecoder.decompress( context.getVideoRawPointsTexture(), path.str(), context.size(), videoBitstreamMP,
                                   params_.videoDecoderPath_, context, decodedBitdepthAttributeMP,
                                   params_.keepIntermediateFiles_, true, false,
                                   params_.inverseColorSpaceConversionConfig_, params_.colorSpaceConversionPath_ );
          context.getVideoTexture().convertBitdepth( decodedBitdepthAttributeMP,
                                                     ai.getAttributeNominal2dBitdepthMinus1( 0 ) + 1,
                                                     ai.getAttributeMSBAlignFlag( 0 ) );
          // generateRawPointsTexturefromVideo( context, reconstructs );
          std::cout << " raw points texture -> " << videoBitstreamMP.size() << " B" << endl;
        }
      }
    }
  }

  // All video have been decoded, start reconsctruction processes
  if ( sps.getRawPatchEnabledFlag( atlasIndex ) && sps.getRawSeparateVideoPresentFlag( atlasIndex ) ) {
    printf("generateRawPointsGeometryfromVideo \n"); fflush(stdout);
    generateRawPointsGeometryfromVideo( context, reconstructs );
  }

  if ( ai.getAttributeCount() > 0 ) {
    for ( int attrIndex = 0; attrIndex < sps.getAttributeInformation( atlasIndex ).getAttributeCount();
          attrIndex++ ) {  // right now we only have one attribute, this should be generalized
      int decodedBitdepthAttribute   = ai.getAttributeNominal2dBitdepthMinus1( attrIndex ) + 1;
      int decodedBitdepthAttributeMP = ai.getAttributeNominal2dBitdepthMinus1( attrIndex ) + 1;
      for ( int attrPartitionIndex = 0;
            attrPartitionIndex <
            sps.getAttributeInformation( atlasIndex ).getAttributeDimensionPartitionsMinus1( attrIndex ) + 1;
            attrPartitionIndex++ ) {  // right now we have only one partition, this should be generalized           
        if ( sps.getRawPatchEnabledFlag( atlasIndex ) && sps.getRawSeparateVideoPresentFlag( atlasIndex ) ) {
          printf("generateRawPointsTexturefromVideo attrIndex = %d attrPartitionIndex = %d \n",attrIndex,attrPartitionIndex); fflush(stdout);
          generateRawPointsTexturefromVideo( context, reconstructs );      
          printf("Done \n"); fflush(stdout);
        }
      }
    }
  }

  reconstructs.setFrameCount( context.size() );
  context.setOccupancyPrecision( sps.getFrameWidth( atlasIndex ) / context.getVideoOccupancyMap().getWidth() );

  printf("setGeneratePointCloudParameters \n"); fflush(stdout);
  GeneratePointCloudParameters gpcParams;
  GeneratePointCloudParameters ppSEIParams;  
  setGeneratePointCloudParameters( gpcParams, context );
  setPostProcessingSeiParameters( ppSEIParams, context );

  // recreating the prediction list per attribute (either the attribute is coded absolute, or follows the geometry)
  // see contribution m52529
  std::vector<std::vector<bool>> absoluteT1List;
  absoluteT1List.resize( ai.getAttributeCount() );
  for ( int attrIdx = 0; attrIdx < ai.getAttributeCount(); ++attrIdx ) {
    absoluteT1List[attrIdx].resize( sps.getMapCountMinus1( atlasIndex ) + 1 );
    if ( ai.getAttributeMapAbsoluteCodingPersistanceFlag( attrIdx ) != 0u ) {
      for ( int mapIdx = 0; mapIdx < sps.getMapCountMinus1( atlasIndex ) + 1; ++mapIdx ) {
        absoluteT1List[attrIdx][mapIdx] = true;
      }
    } else {
      // follow geometry
      for ( int mapIdx = 0; mapIdx < sps.getMapCountMinus1( atlasIndex ) + 1; ++mapIdx ) {
        absoluteT1List[attrIdx][mapIdx] = sps.getMapAbsoluteCodingEnableFlag( atlasIndex, mapIdx );
      }
    }
  }

  for ( auto& frame : context.getFrames() ) {
    auto& reconstruct = reconstructs[ frame.getIndex() ]; 
    std::vector<uint32_t> partition;

    // Decode point cloud
    generateOccupancyMap( frame, context.getVideoOccupancyMap().getFrame( frame.getIndex() ), 
                          context.getOccupancyPrecision(), oi.getLossyOccupancyMapCompressionThreshold(),
                          asps.getEnhancedOccupancyMapForDepthFlag());
    generateBlockToPatchFromBoundaryBox( context, frame, context.getOccupancyPackingBlockSize()  );
    generatePointCloud( reconstruct, context, frame, gpcParams, partition, true );
    for ( size_t attIdx = 0; attIdx < ai.getAttributeCount(); attIdx++ ) {
      colorPointCloud( reconstruct, context, frame, absoluteT1List[attIdx],  
                       sps.getMultipleMapStreamsPresentFlag( ATLASIDXPCC ), ai.getAttributeCount(), 
                       gpcParams );
    }

    // Post-Processing
    use444CodecIo = sps.getRawPatchEnabledFlag( 0 ); 
    // use444CodecIo = sps.getRawPatchEnabledFlag( 0 ) && plt.getProfileCodecGroupIdc() == CODEC_GROUP_HEVC444;
    // use444CodecIo = plt.getProfileCodecGroupIdc() == CODEC_GROUP_HEVC444;
    if ( ppSEIParams.flagGeometrySmoothing_ ) {
      PCCPointSet3 tempFrameBuffer = reconstruct;
      if ( ppSEIParams.gridSmoothing_ ) {
        smoothPointCloudPostprocess( reconstruct, context, params_.colorTransform_, ppSEIParams, partition );
      }
      // These are different attribute transfer functions
      if ( params_.postprocessSmoothingFilter_ == 1 ) {
        tempFrameBuffer.transferColors16bitBP( reconstruct, int32_t( 0 ), use444CodecIo, 8, 1, true, true, true,
                                               false, 4, 4, 1000, 1000, 1000 * 256,
                                               1000 * 256 );  // jkie: let's make it general
      } else if ( params_.postprocessSmoothingFilter_ == 2 ) {
        tempFrameBuffer.transferColorWeight( reconstruct, 0.1 );
      } else if ( params_.postprocessSmoothingFilter_ == 3 ) {
        tempFrameBuffer.transferColorsFilter3( reconstruct, int32_t( 0 ), use444CodecIo );
      }
    }
    if ( ppSEIParams.flagColorSmoothing_ ) {
      colorSmoothing( reconstruct, context, params_.colorTransform_, ppSEIParams );
    }
    if ( !use444CodecIo ) {  // lossy: convert 16-bit yuv444 to 8-bit RGB444
      printf( " convert 16-bit yuv444 to 8-bit RGB444 \n" );
      reconstruct.convertYUV16ToRGB8();
    } else {  // lossless: copy 16-bit RGB to 8-bit RGB
      printf( "lossless: copy 16-bit RGB to 8-bit RGB\n" );
      reconstruct.copyRGB16ToRGB8();     
    }
  }
#ifdef CODEC_TRACE
  setTrace( false );
  closeTrace();
#endif
  return 0;
}

void PCCDecoder::setPointLocalReconstruction( PCCContext& context ) {
  auto&                        asps = context.getAtlasSequenceParameterSet( 0 );
  PointLocalReconstructionMode mode = {false, false, 0, 1};
  context.addPointLocalReconstructionMode( mode );
  if ( asps.getPointLocalReconstructionEnabledFlag() ) {
    auto& plri = asps.getPointLocalReconstructionInformation( 0 );
    for ( size_t i = 0; i < plri.getNumberOfModesMinus1(); i++ ) {
      mode.interpolate_ = plri.getInterpolateFlag( i );
      mode.filling_     = plri.getFillingFlag( i );
      mode.minD1_       = plri.getMinimumDepth( i );
      mode.neighbor_    = plri.getNeighbourMinus1( i ) + 1;
      context.addPointLocalReconstructionMode( mode );
    }
  }
#ifdef CODEC_TRACE
  for ( size_t i = 0; i < context.getPointLocalReconstructionModeNumber(); i++ ) {
    auto& mode = context.getPointLocalReconstructionMode( i );
    TRACE_CODEC( "Plrm[%zu]: Inter = %d Fill = %d minD1 = %u neighbor = %u \n", i, mode.interpolate_, mode.filling_,
                 mode.minD1_, mode.neighbor_ );
  }
#endif
}

void PCCDecoder::setPointLocalReconstructionData( PCCFrameContext&              frame,
                                                  PCCPatch&                     patch,
                                                  PointLocalReconstructionData& plrd,
                                                  size_t                        occupancyPackingBlockSize ) {
  patch.allocOneLayerData();
  TRACE_CODEC( "WxH = %zu x %zu \n", plrd.getBlockToPatchMapWidth(), plrd.getBlockToPatchMapHeight() );
  patch.getPointLocalReconstructionLevel() = static_cast<uint8_t>( plrd.getLevelFlag() );
  TRACE_CODEC( "  LevelFlag = %d \n", plrd.getLevelFlag() );
  if ( plrd.getLevelFlag() ) {
    if ( plrd.getPresentFlag() ) {
      patch.getPointLocalReconstructionMode() = plrd.getModeMinus1() + 1;
    } else {
      patch.getPointLocalReconstructionMode() = 0;
    }
    TRACE_CODEC( "  ModePatch: Present = %d ModeMinus1 = %2d \n", plrd.getPresentFlag(),
                 plrd.getPresentFlag() ? (int32_t)plrd.getModeMinus1() : -1 );
  } else {
    for ( size_t v0 = 0; v0 < plrd.getBlockToPatchMapHeight(); ++v0 ) {
      for ( size_t u0 = 0; u0 < plrd.getBlockToPatchMapWidth(); ++u0 ) {
        size_t index = v0 * plrd.getBlockToPatchMapWidth() + u0;
        if ( plrd.getBlockPresentFlag( index ) ) {
          patch.getPointLocalReconstructionMode( u0, v0 ) = plrd.getBlockModeMinus1( index ) + 1;
        } else {
          patch.getPointLocalReconstructionMode( u0, v0 ) = 0;
        }
        TRACE_CODEC( "  Mode[%3u]: Present = %d ModeMinus1 = %2d \n", index, plrd.getBlockPresentFlag( index ),
                     plrd.getBlockPresentFlag( index ) ? (int32_t)plrd.getBlockModeMinus1( index ) : -1 );
      }
    }
  }
#ifdef CODEC_TRACE
  for ( size_t v0 = 0; v0 < patch.getSizeV0(); ++v0 ) {
    for ( size_t u0 = 0; u0 < patch.getSizeU0(); ++u0 ) {
      TRACE_CODEC(
          "Block[ %2lu %2lu <=> %4zu ] / [ %2lu %2lu ]: Level = %d Present = "
          "%d mode = %zu \n",
          u0, v0, v0 * patch.getSizeU0() + u0, patch.getSizeU0(), patch.getSizeV0(),
          patch.getPointLocalReconstructionLevel(), plrd.getBlockPresentFlag( v0 * patch.getSizeU0() + u0 ),
          patch.getPointLocalReconstructionMode( u0, v0 ) );
    }
  }
#endif
}

void PCCDecoder::setPostProcessingSeiParameters( GeneratePointCloudParameters& params,
                                                 PCCContext&                   context ) {
  auto&   sps                   = context.getVps();
  int32_t atlasIndex            = 0;
  auto&   oi                    = sps.getOccupancyInformation( atlasIndex );
  auto&   gi                    = sps.getGeometryInformation( atlasIndex );
  auto&   asps                  = context.getAtlasSequenceParameterSet( 0 );
  bool    seiSmoothingIsPresent = context.seiIsPresent( NAL_PREFIX_SEI, SMOOTHING_PARAMETERS );
  auto&   plt                   = sps.getProfileTierLevel();
  params.flagGeometrySmoothing_ = false;
  params.gridSmoothing_         = false;
  params.gridSize_              = 0;
  params.thresholdSmoothing_    = 0;
  params.pbfEnableFlag_         = false;
  params.pbfPassesCount_        = 0;
  params.pbfFilterSize_         = 0;
  params.pbfLog2Threshold_      = 0;
  if ( seiSmoothingIsPresent ) {
    auto* sei = static_cast<SEISmoothingParameters*>( context.getSei( NAL_PREFIX_SEI, SMOOTHING_PARAMETERS ) );
    if ( sei->getSpGeometrySmoothingEnabledFlag() ) {
      params.flagGeometrySmoothing_ = true;
      if ( sei->getSpGeometrySmoothingId() == 0 ) {
        params.gridSmoothing_      = true;
        params.gridSize_           = sei->getSpGeometrySmoothingGridSizeMinus2() + 2;
        params.thresholdSmoothing_ = static_cast<double>( sei->getSpGeometrySmoothingThreshold() );
      }
      if ( sei->getSpGeometrySmoothingId() == 1 ) {
        params.pbfEnableFlag_    = true;
        params.pbfPassesCount_   = sei->getSpGeometryPatchBlockFilteringPassesCountMinus1() + 1;
        params.pbfFilterSize_    = sei->getSpGeometryPatchBlockFilteringFilterSizeMinus1() + 1;
        params.pbfLog2Threshold_ = sei->getSpGeometryPatchBlockFilteringLog2ThresholdMinus1() + 1;
      }
    }
  }
  params.occupancyResolution_    = context.getOccupancyPackingBlockSize();
  params.occupancyPrecision_     = context.getOccupancyPrecision();
  params.enableSizeQuantization_ = context.getAtlasSequenceParameterSet( 0 ).getPatchSizeQuantizerPresentFlag();
  params.rawPointColorFormat_ =
      size_t( plt.getProfileCodecGroupIdc() == CODEC_GROUP_HEVC444 ? COLOURFORMAT444 : COLOURFORMAT420 );
  params.nbThread_   = params_.nbThread_;
  params.absoluteD1_ = sps.getMapCountMinus1( atlasIndex ) == 0 || sps.getMapAbsoluteCodingEnableFlag( atlasIndex, 1 );
  params.multipleStreams_             = sps.getMultipleMapStreamsPresentFlag( atlasIndex );
  params.surfaceThickness_            = asps.getSurfaceThicknessMinus1() + 1;
  params.thresholdColorSmoothing_     = 0.;
  params.gridColorSmoothing_          = false;
  params.cgridSize_                   = 0;
  params.thresholdColorDifference_    = 0;
  params.thresholdColorVariation_     = 0;
  params.thresholdLocalEntropy_       = 0;
  params.radius2ColorSmoothing_       = 64;
  params.neighborCountColorSmoothing_ = 64;
  params.flagColorSmoothing_          = false;
  if ( seiSmoothingIsPresent ) {
    auto* sei = static_cast<SEISmoothingParameters*>( context.getSei( NAL_PREFIX_SEI, SMOOTHING_PARAMETERS ) );
    for ( size_t j = 0; j < sei->getSpNumAttributeUpdates(); j++ ) {
      size_t index = sei->getSpAttributeIdx( j );
      for ( size_t i = 0; i < sei->getSpDimensionMinus1( index ) + 1; i++ ) {
        if ( sei->getSpAttrSmoothingParamsEnabledFlag( index, i ) ) {
          params.flagColorSmoothing_       = true;
          params.thresholdColorSmoothing_  = static_cast<double>( sei->getSpAttrSmoothingThreshold( index, i ) );
          params.gridColorSmoothing_       = true;
          params.cgridSize_                = sei->getSpAttrSmoothingGridSizeMinus2( index, i ) + 2;
          params.thresholdColorDifference_ = sei->getSpAttrSmoothingThresholdDifference( index, i );
          params.thresholdColorVariation_  = sei->getSpAttrSmoothingThresholdVariation( index, i );
          params.thresholdLocalEntropy_    = sei->getSpAttrSmoothingLocalEntropyThreshold( index, i );
        }
      }
    }
  }
  params.thresholdLossyOM_              = static_cast<size_t>( oi.getLossyOccupancyMapCompressionThreshold() );
  params.removeDuplicatePoints_         = asps.getRemoveDuplicatePointEnabledFlag();
  params.pointLocalReconstruction_      = asps.getPointLocalReconstructionEnabledFlag();
  params.mapCountMinus1_                = sps.getMapCountMinus1( atlasIndex );
  params.singleMapPixelInterleaving_    = asps.getPixelDeinterleavingFlag();
  params.useAdditionalPointsPatch_      = sps.getRawPatchEnabledFlag( atlasIndex );
  params.enhancedOccupancyMapCode_      = asps.getEnhancedOccupancyMapForDepthFlag();
  params.EOMFixBitCount_                = asps.getEnhancedOccupancyMapFixBitCountMinus1() + 1;
  params.geometry3dCoordinatesBitdepth_ = gi.getGeometry3dCoordinatesBitdepthMinus1() + 1;
  params.geometryBitDepth3D_            = gi.getGeometry3dCoordinatesBitdepthMinus1() + 1;
}

void PCCDecoder::setGeneratePointCloudParameters( GeneratePointCloudParameters& params, PCCContext& context ) {
  auto&   sps                   = context.getVps();
  int32_t atlasIndex            = 0;
  auto&   oi                    = sps.getOccupancyInformation( atlasIndex );
  auto&   gi                    = sps.getGeometryInformation( atlasIndex );
  auto&   asps                  = context.getAtlasSequenceParameterSet( 0 );
  bool    seiSmoothingIsPresent = context.seiIsPresent( NAL_PREFIX_SEI, SMOOTHING_PARAMETERS );
  auto&   plt                   = sps.getProfileTierLevel();
  params.flagGeometrySmoothing_ = false;
  params.gridSmoothing_         = false;
  params.gridSize_              = 0;
  params.thresholdSmoothing_    = 0;
  params.pbfEnableFlag_         = false;
  params.pbfPassesCount_        = 0;
  params.pbfFilterSize_         = 0;
  params.pbfLog2Threshold_      = 0;
  if ( seiSmoothingIsPresent ) {
    auto* sei = static_cast<SEISmoothingParameters*>( context.getSei( NAL_PREFIX_SEI, SMOOTHING_PARAMETERS ) );

    if ( sei->getSpGeometrySmoothingEnabledFlag() ) {
      params.flagGeometrySmoothing_ = true;
      if ( sei->getSpGeometrySmoothingId() == 0 ) {
        params.gridSmoothing_      = true;
        params.gridSize_           = sei->getSpGeometrySmoothingGridSizeMinus2() + 2;
        params.thresholdSmoothing_ = static_cast<double>( sei->getSpGeometrySmoothingThreshold() );
      }
      if ( sei->getSpGeometrySmoothingId() == 1 ) {
        params.pbfEnableFlag_    = true;
        params.pbfPassesCount_   = sei->getSpGeometryPatchBlockFilteringPassesCountMinus1() + 1;
        params.pbfFilterSize_    = sei->getSpGeometryPatchBlockFilteringFilterSizeMinus1() + 1;
        params.pbfLog2Threshold_ = sei->getSpGeometryPatchBlockFilteringLog2ThresholdMinus1() + 1;
      }
    }
  }
  params.occupancyResolution_    = context.getOccupancyPackingBlockSize();
  params.occupancyPrecision_     = context.getOccupancyPrecision();
  params.enableSizeQuantization_ = context.getAtlasSequenceParameterSet( 0 ).getPatchSizeQuantizerPresentFlag();
  params.rawPointColorFormat_ =
      size_t( plt.getProfileCodecGroupIdc() == CODEC_GROUP_HEVC444 ? COLOURFORMAT444 : COLOURFORMAT420 );
  params.nbThread_   = params_.nbThread_;
  params.absoluteD1_ = sps.getMapCountMinus1( atlasIndex ) == 0 || sps.getMapAbsoluteCodingEnableFlag( atlasIndex, 1 );
  params.multipleStreams_             = sps.getMultipleMapStreamsPresentFlag( atlasIndex );
  params.surfaceThickness_            = asps.getSurfaceThicknessMinus1() + 1;
  params.thresholdColorSmoothing_     = 0.;
  params.gridColorSmoothing_          = false;
  params.cgridSize_                   = 0;
  params.thresholdColorDifference_    = 0;
  params.thresholdColorVariation_     = 0;
  params.thresholdLocalEntropy_       = 0;
  params.radius2ColorSmoothing_       = 64;
  params.neighborCountColorSmoothing_ = 64;
  params.flagColorSmoothing_          = false;
  if ( seiSmoothingIsPresent ) {
    auto* sei = static_cast<SEISmoothingParameters*>( context.getSei( NAL_PREFIX_SEI, SMOOTHING_PARAMETERS ) );
    for ( size_t j = 0; j < sei->getSpNumAttributeUpdates(); j++ ) {
      size_t index = sei->getSpAttributeIdx( j );
      for ( size_t i = 0; i < sei->getSpDimensionMinus1( index ) + 1; i++ ) {
        if ( sei->getSpAttrSmoothingParamsEnabledFlag( index, i ) ) {
          params.flagColorSmoothing_       = true;
          params.thresholdColorSmoothing_  = static_cast<double>( sei->getSpAttrSmoothingThreshold( index, i ) );
          params.gridColorSmoothing_       = true;
          params.cgridSize_                = sei->getSpAttrSmoothingGridSizeMinus2( index, i ) + 2;
          params.thresholdColorDifference_ = sei->getSpAttrSmoothingThresholdDifference( index, i );
          params.thresholdColorVariation_  = sei->getSpAttrSmoothingThresholdVariation( index, i );
          params.thresholdLocalEntropy_    = sei->getSpAttrSmoothingLocalEntropyThreshold( index, i );
        }
      }
    }
  }
  params.thresholdLossyOM_              = static_cast<size_t>( oi.getLossyOccupancyMapCompressionThreshold() );
  params.removeDuplicatePoints_         = asps.getRemoveDuplicatePointEnabledFlag();
  params.pointLocalReconstruction_      = asps.getPointLocalReconstructionEnabledFlag();
  params.mapCountMinus1_                = sps.getMapCountMinus1( atlasIndex );
  params.singleMapPixelInterleaving_    = asps.getPixelDeinterleavingFlag();
  params.useAdditionalPointsPatch_      = sps.getRawPatchEnabledFlag( atlasIndex );
  params.enhancedOccupancyMapCode_      = asps.getEnhancedOccupancyMapForDepthFlag();
  params.EOMFixBitCount_                = asps.getEnhancedOccupancyMapFixBitCountMinus1() + 1;
  params.geometry3dCoordinatesBitdepth_ = gi.getGeometry3dCoordinatesBitdepthMinus1() + 1;
  params.geometryBitDepth3D_            = gi.getGeometry3dCoordinatesBitdepthMinus1() + 1;
}

void PCCDecoder::createPatchFrameDataStructure( PCCContext& context ) {
  TRACE_CODEC( "createPatchFrameDataStructure GOP start \n" );
  auto&  sps        = context.getVps();
  size_t atlasIndex = context.getAtlasIndex();
  // auto&  asps       = context.getAtlasSequenceParameterSet( /*0*/ );
  auto& atglulist = context.getAtlasTileGroupLayerList();
  // context.setOccupancyPackingBlockSize( pow( 2,
  // asps.getLog2PatchPackingBlockSize() ) );
  context.resize( atglulist.size() );  // assuming that we have just one tile
                                       // group per frame, how should we get the
                                       // total number of frames?
  TRACE_CODEC( "frameCount = %u \n", context.size() );
  setPointLocalReconstruction( context );
  context.constructRefList( 0, 0 );
  context.setMPGeoWidth( 64 );
  context.setMPAttWidth( 0 );
  context.setMPGeoHeight( 0 );
  context.setMPAttHeight( 0 );

  for ( size_t i = 0; i < context.size(); i++ ) {
    auto& frame = context.getFrame( i );
    if ( i > 0 ) { frame.setRefAFOCList( context ); }
    frame.setAFOC( i );
    frame.setIndex( i );
    frame.setWidth( sps.getFrameWidth( atlasIndex ) );
    frame.setHeight( sps.getFrameHeight( atlasIndex ) );
    frame.setUseRawPointsSeparateVideo( sps.getRawSeparateVideoPresentFlag( atlasIndex ) );
    frame.setRawPatchEnabledFlag( sps.getRawPatchEnabledFlag( atlasIndex ) );
    auto& afps = context.getAtlasFrameParameterSet();  // how to determine which
                                                       // AFPS is valid for a
                                                       // particular POC?
    // Right now we only have one anyway
    size_t numTileGroups = atglulist[i].size();
    createPatchFrameDataStructure( context, frame, i );
  }
}
void PCCDecoder::createPatchFrameDataStructure( PCCContext& context, PCCFrameContext& frame, size_t frameIndex ) {
  TRACE_CODEC( "createPatchFrameDataStructure Frame %zu \n", frame.getIndex() );
  auto&    sps                  = context.getVps();
  size_t   atlasIndex           = context.getAtlasIndex();
  auto&    gi                   = sps.getGeometryInformation( atlasIndex );
  uint32_t NumTilesInPatchFrame = 1;  // (afti.getNumTileColumnsMinus1() + 1) *
                                      // (afti.getNumTileRowsMinus1() + 1);
  // TODO: How to determine the number of tiles in a frame??? Currently one tile
  // is used, but if multiple tiles are
  // used, the code below is wrong (patches are not accumulated, but overwriten)
  for ( size_t tileGroupIndex = 0; tileGroupIndex < NumTilesInPatchFrame; tileGroupIndex++ ) {
    auto& atglu = context.getAtlasTileGroupLayer( frameIndex, tileGroupIndex );
    auto& atgh  = atglu.getAtlasTileGroupHeader();
    // the header indicates the structures used
    auto& afps  = context.getAtlasFrameParameterSet( atgh.getAtghAtlasFrameParameterSetId() );
    auto& asps  = context.getAtlasSequenceParameterSet( afps.getAtlasSequenceParameterSetId() );
    auto& atgdu = atglu.getAtlasTileGroupDataUnit();

    // local variable initialization
    auto&        patches                 = frame.getPatches();
    auto&        pcmPatches              = frame.getRawPointsPatches();
    auto&        eomPatches              = frame.getEomPatches();
    int64_t      prevSizeU0              = 0;
    int64_t      prevSizeV0              = 0;
    int64_t      prevPatchSize2DXInPixel = 0;
    int64_t      prevPatchSize2DYInPixel = 0;
    int64_t      predIndex               = 0;
    const size_t minLevel                = pow( 2., atgh.getAtghPosMinZQuantizer() );
    size_t       numRawPatches           = 0;
    size_t       numNonRawPatch          = 0;
    size_t       numEomPatch             = 0;
    PCCTILEGROUP tileGroupType           = atgh.getAtghType();
    size_t       patchCount              = atgdu.getPatchCount();
    for ( size_t i = 0; i < patchCount; i++ ) {
      PCCPatchType currPatchType = getCurrPatchType( tileGroupType, atgdu.getPatchMode( i ) );
      if ( currPatchType == RAW_PATCH ) {
        numRawPatches++;
      } else if ( currPatchType == EOM_PATCH ) {
        numEomPatch++;
      }
    }
    numNonRawPatch = patchCount - numRawPatches - numEomPatch;
    eomPatches.reserve( numEomPatch );
    patches.resize( numNonRawPatch );
    pcmPatches.resize( numRawPatches );
    TRACE_CODEC( "Patches size                        = %zu \n", patches.size() );
    TRACE_CODEC( "non-regular Patches(pcm, eom)     = %zu, %zu \n", numRawPatches, numEomPatch );
    TRACE_CODEC(
        "TileGroup Type                     = %zu (0.P_TILE_GRP "
        "1.SKIP_TILE_GRP 2.I_TILE_GRP)\n",
        (size_t)atgh.getAtghType() );
    TRACE_CODEC( "OccupancyPackingBlockSize           = %d \n", context.getOccupancyPackingBlockSize() );
    size_t  totalNumberOfRawPoints = 0;
    size_t  patchIndex             = 0;
    int32_t packingBlockSize       = context.getOccupancyPackingBlockSize();
    int32_t quantizerSizeX         = 1 << atgh.getAtghPatchSizeXinfoQuantizer();
    int32_t quantizerSizeY         = 1 << atgh.getAtghPatchSizeYinfoQuantizer();
    frame.setLog2PatchQuantizerSizeX( atgh.getAtghPatchSizeXinfoQuantizer() );
    frame.setLog2PatchQuantizerSizeY( atgh.getAtghPatchSizeYinfoQuantizer() );
    for ( patchIndex = 0; patchIndex < patchCount; patchIndex++ ) {
      auto&        pid           = atgdu.getPatchInformationData( patchIndex );
      PCCPatchType currPatchType = getCurrPatchType( tileGroupType, atgdu.getPatchMode( patchIndex ) );
      if ( currPatchType == INTRA_PATCH ) {
        auto& patch                    = patches[patchIndex];
        patch.getOccupancyResolution() = context.getOccupancyPackingBlockSize();
        auto& pdu                      = pid.getPatchDataUnit();
        patch.getU0()                  = pdu.getPdu2dPosX();
        patch.getV0()                  = pdu.getPdu2dPosY();
        patch.getU1()                  = pdu.getPdu3dPosX();
        patch.getV1()                  = pdu.getPdu3dPosY();

        bool lodEnableFlag = pdu.getLodEnableFlag();
        if ( lodEnableFlag ) {
          patch.setLodScaleX( pdu.getLodScaleXminus1() + 1 );
          patch.setLodScaleY( pdu.getLodScaleY() + ( patch.getLodScaleX() > 1 ? 1 : 2 ) );
        } else {
          patch.setLodScaleX( 1 );
          patch.setLodScaleY( 1 );
        }
        patch.getSizeD() = ( std::min )( pdu.getPdu3dPosDeltaMaxZ() * minLevel, (size_t)255 );
        if ( asps.getPatchSizeQuantizerPresentFlag() ) {
          patch.setPatchSize2DXInPixel( pdu.getPdu2dSizeXMinus1() * quantizerSizeX + 1 );
          patch.setPatchSize2DYInPixel( pdu.getPdu2dSizeYMinus1() * quantizerSizeY + 1 );
          patch.getSizeU0() =
              ceil( static_cast<double>( patch.getPatchSize2DXInPixel() ) / static_cast<double>( packingBlockSize ) );
          patch.getSizeV0() =
              ceil( static_cast<double>( patch.getPatchSize2DYInPixel() ) / static_cast<double>( packingBlockSize ) );
        } else {
          patch.getSizeU0() = pdu.getPdu2dSizeXMinus1() + 1;
          patch.getSizeV0() = pdu.getPdu2dSizeYMinus1() + 1;
        }
        size_t pduProjectionPlane =
            asps.get45DegreeProjectionPatchPresentFlag() ? ( pdu.getPduProjectionId() >> 2 ) : pdu.getPduProjectionId();
        size_t pdu45degreeProjectionRotationAxis =
            asps.get45DegreeProjectionPatchPresentFlag() ? ( pdu.getPduProjectionId() & 0x03 ) : 0;
        patch.getNormalAxis()            = pduProjectionPlane % 3;
        patch.getProjectionMode()        = pduProjectionPlane < 3 ? 0 : 1;
        patch.getPatchOrientation()      = pdu.getPduOrientationIndex();
        patch.getAxisOfAdditionalPlane() = pdu45degreeProjectionRotationAxis;
        TRACE_CODEC( "patch %zu / %zu: Intra \n", patchIndex, patchCount );
        const size_t max3DCoordinate = size_t( 1 ) << ( gi.getGeometry3dCoordinatesBitdepthMinus1() + 1 );
        if ( patch.getProjectionMode() == 0 ) {
          patch.getD1() = static_cast<int32_t>( pdu.getPdu3dPosMinZ() ) * minLevel;
        } else {
          if ( static_cast<int>( asps.get45DegreeProjectionPatchPresentFlag() ) == 0 ) {
            patch.getD1() = max3DCoordinate - static_cast<int32_t>( pdu.getPdu3dPosMinZ() ) * minLevel;
          } else {
            patch.getD1() = ( max3DCoordinate << 1 ) - static_cast<int32_t>( pdu.getPdu3dPosMinZ() ) * minLevel;
          }
        }
        prevSizeU0              = patch.getSizeU0();
        prevSizeV0              = patch.getSizeV0();
        prevPatchSize2DXInPixel = patch.getPatchSize2DXInPixel();
        prevPatchSize2DYInPixel = patch.getPatchSize2DYInPixel();
        if ( patch.getNormalAxis() == 0 ) {
          patch.getTangentAxis()   = 2;
          patch.getBitangentAxis() = 1;
        } else if ( patch.getNormalAxis() == 1 ) {
          patch.getTangentAxis()   = 2;
          patch.getBitangentAxis() = 0;
        } else {
          patch.getTangentAxis()   = 0;
          patch.getBitangentAxis() = 1;
        }
        TRACE_CODEC(
            "patch(Intra) %zu: UV0 %4zu %4zu UV1 %4zu %4zu D1=%4zu S=%4zu %4zu "
            "%4zu(%4zu) P=%zu O=%zu A=%u%u%u Lod "
            "=(%zu) %zu,%zu 45=%d ProjId=%4zu Axis=%zu \n",
            patchIndex, patch.getU0(), patch.getV0(), patch.getU1(), patch.getV1(), patch.getD1(), patch.getSizeU0(),
            patch.getSizeV0(), patch.getSizeD(), pdu.getPdu3dPosDeltaMaxZ(), patch.getProjectionMode(),
            patch.getPatchOrientation(), patch.getNormalAxis(), patch.getTangentAxis(), patch.getBitangentAxis(),
            (size_t)lodEnableFlag, patch.getLodScaleX(), patch.getLodScaleY(),
            asps.get45DegreeProjectionPatchPresentFlag(), pdu.getPduProjectionId(), patch.getAxisOfAdditionalPlane() );
        patch.allocOneLayerData();
        if ( asps.getPointLocalReconstructionEnabledFlag() ) {
          setPointLocalReconstructionData( frame, patch, pdu.getPointLocalReconstructionData(),
                                           context.getOccupancyPackingBlockSize() );
        }
      } else if ( currPatchType == INTER_PATCH ) {
        auto& patch                    = patches[patchIndex];
        patch.getOccupancyResolution() = context.getOccupancyPackingBlockSize();
        auto& ipdu                     = pid.getInterPatchDataUnit();

        TRACE_CODEC( "patch %zu / %zu: Inter \n", patchIndex, patchCount );
        TRACE_CODEC(
            "IPDU: refAtlasFrame= %d refPatchIdx = %d pos2DXY = %ld %ld "
            "pos3DXYZW = %ld %ld %ld %ld size2D = %ld %ld "
            "\n",
            ipdu.getIpduRefIndex(), ipdu.getIpduRefPatchIndex(), ipdu.getIpdu2dPosX(), ipdu.getIpdu2dPosY(),
            ipdu.getIpdu3dPosX(), ipdu.getIpdu3dPosY(), ipdu.getIpdu3dPosMinZ(), ipdu.getIpdu3dPosDeltaMaxZ(),
            ipdu.getIpdu2dDeltaSizeX(), ipdu.getIpdu2dDeltaSizeY() );
        patch.setBestMatchIdx( static_cast<int32_t>( ipdu.getIpduRefPatchIndex() + predIndex ) );
        predIndex += ipdu.getIpduRefPatchIndex() + 1;
        patch.setRefAtlasFrameIndex( ipdu.getIpduRefIndex() );
        size_t      refPOC   = frame.getRefAFOC( patch.getRefAtlasFrameIndex() );
        const auto& refPatch = context.getFrame( refPOC ).getPatches()[patch.getBestMatchIdx()];
        TRACE_CODEC(
            "\trefPatch: Idx = %zu UV0 = %zu %zu  UV1 = %zu %zu Size = %zu %zu "
            "%zu  Lod = %u,%u\n",
            patch.getBestMatchIdx(), refPatch.getU0(), refPatch.getV0(), refPatch.getU1(), refPatch.getV1(),
            refPatch.getSizeU0(), refPatch.getSizeV0(), refPatch.getSizeD(), refPatch.getLodScaleX(),
            refPatch.getLodScaleY() );
        patch.getProjectionMode()   = refPatch.getProjectionMode();
        patch.getU0()               = ipdu.getIpdu2dPosX() + refPatch.getU0();
        patch.getV0()               = ipdu.getIpdu2dPosY() + refPatch.getV0();
        patch.getPatchOrientation() = refPatch.getPatchOrientation();
        patch.getU1()               = ipdu.getIpdu3dPosX() + refPatch.getU1();
        patch.getV1()               = ipdu.getIpdu3dPosY() + refPatch.getV1();
        if ( asps.getPatchSizeQuantizerPresentFlag() ) {
          patch.setPatchSize2DXInPixel( refPatch.getPatchSize2DXInPixel() +
                                        ( ipdu.getIpdu2dDeltaSizeX() ) * quantizerSizeX );
          patch.setPatchSize2DYInPixel( refPatch.getPatchSize2DYInPixel() +
                                        ( ipdu.getIpdu2dDeltaSizeY() ) * quantizerSizeY );
          patch.getSizeU0() =
              ceil( static_cast<double>( patch.getPatchSize2DXInPixel() ) / static_cast<double>( packingBlockSize ) );
          patch.getSizeV0() =
              ceil( static_cast<double>( patch.getPatchSize2DYInPixel() ) / static_cast<double>( packingBlockSize ) );
        } else {
          patch.getSizeU0() = ipdu.getIpdu2dDeltaSizeX() + refPatch.getSizeU0();
          patch.getSizeV0() = ipdu.getIpdu2dDeltaSizeY() + refPatch.getSizeV0();
        }
        patch.getNormalAxis()            = refPatch.getNormalAxis();
        patch.getTangentAxis()           = refPatch.getTangentAxis();
        patch.getBitangentAxis()         = refPatch.getBitangentAxis();
        patch.getAxisOfAdditionalPlane() = refPatch.getAxisOfAdditionalPlane();
        const size_t max3DCoordinate     = size_t( 1 ) << ( gi.getGeometry3dCoordinatesBitdepthMinus1() + 1 );
        if ( patch.getProjectionMode() == 0 ) {
          patch.getD1() = ( ipdu.getIpdu3dPosMinZ() + ( refPatch.getD1() / minLevel ) ) * minLevel;
        } else {
          if ( static_cast<int>( asps.get45DegreeProjectionPatchPresentFlag() ) == 0 ) {
            patch.getD1() =
                max3DCoordinate -
                ( ipdu.getIpdu3dPosMinZ() + ( ( max3DCoordinate - refPatch.getD1() ) / minLevel ) ) * minLevel;
          } else {
            patch.getD1() =
                ( max3DCoordinate << 1 ) -
                ( ipdu.getIpdu3dPosMinZ() + ( ( ( max3DCoordinate << 1 ) - refPatch.getD1() ) / minLevel ) ) * minLevel;
          }
        }
        const int64_t delta_DD = ipdu.getIpdu3dPosDeltaMaxZ();
        size_t        prevDD   = refPatch.getSizeD() / minLevel;
        if ( prevDD * minLevel != refPatch.getSizeD() ) { prevDD += 1; }
        patch.getSizeD() = ( std::min )( size_t( ( delta_DD + prevDD ) * minLevel ), (size_t)255 );
        patch.setLodScaleX( refPatch.getLodScaleX() );
        patch.setLodScaleY( refPatch.getLodScaleY() );
        prevSizeU0              = patch.getSizeU0();
        prevSizeV0              = patch.getSizeV0();
        prevPatchSize2DXInPixel = patch.getPatchSize2DXInPixel();
        prevPatchSize2DYInPixel = patch.getPatchSize2DYInPixel();

        TRACE_CODEC(
            "patch(Inter) %zu: UV0 %4zu %4zu UV1 %4zu %4zu D1=%4zu S=%4zu %4zu "
            "%4zu from DeltaSize = "
            "%4ld %4ld P=%zu O=%zu A=%u%u%u Lod = %zu,%zu \n",
            patchIndex, patch.getU0(), patch.getV0(), patch.getU1(), patch.getV1(), patch.getD1(), patch.getSizeU0(),
            patch.getSizeV0(), patch.getSizeD(), ipdu.getIpdu2dDeltaSizeX(), ipdu.getIpdu2dDeltaSizeY(),
            patch.getProjectionMode(), patch.getPatchOrientation(), patch.getNormalAxis(), patch.getTangentAxis(),
            patch.getBitangentAxis(), patch.getLodScaleX(), patch.getLodScaleY() );

        patch.allocOneLayerData();
        if ( asps.getPointLocalReconstructionEnabledFlag() ) {
          setPointLocalReconstructionData( frame, patch, ipdu.getPointLocalReconstructionData(),
                                           context.getOccupancyPackingBlockSize() );
        }
      } else if ( currPatchType == MERGE_PATCH ) {
        assert( -2 );
        auto& patch                    = patches[patchIndex];
        patch.getOccupancyResolution() = context.getOccupancyPackingBlockSize();
        auto&        mpdu              = pid.getMergePatchDataUnit();
        bool         overridePlrFlag   = false;
        const size_t max3DCoordinate   = size_t( 1 ) << ( gi.getGeometry3dCoordinatesBitdepthMinus1() + 1 );

        TRACE_CODEC( "patch %zu / %zu: Inter \n", patchIndex, patchCount );
        TRACE_CODEC(
            "MPDU: refAtlasFrame= %d refPatchIdx = ?? pos2DXY = %ld %ld "
            "pos3DXYZW = %ld %ld %ld %ld size2D = %ld %ld "
            "\n",
            mpdu.getMpduRefIndex(), mpdu.getMpdu2dPosX(), mpdu.getMpdu2dPosY(), mpdu.getMpdu3dPosX(),
            mpdu.getMpdu3dPosY(), mpdu.getMpdu3dPosMinZ(), mpdu.getMpdu3dPosDeltaMaxZ(), mpdu.getMpdu2dDeltaSizeX(),
            mpdu.getMpdu2dDeltaSizeY() );

        patch.setBestMatchIdx( patchIndex );
        predIndex = patchIndex;
        patch.setRefAtlasFrameIndex( mpdu.getMpduRefIndex() );
        size_t      refPOC   = frame.getRefAFOC( patch.getRefAtlasFrameIndex() );
        const auto& refPatch = context.getFrame( refPOC ).getPatches()[patch.getBestMatchIdx()];

        if ( mpdu.getMpduOverride2dParamsFlag() ) {
          patch.getU0() = mpdu.getMpdu2dPosX() + refPatch.getU0();
          patch.getV0() = mpdu.getMpdu2dPosY() + refPatch.getV0();
          if ( asps.getPatchSizeQuantizerPresentFlag() ) {
            patch.setPatchSize2DXInPixel( refPatch.getPatchSize2DXInPixel() +
                                          ( mpdu.getMpdu2dDeltaSizeX() ) * quantizerSizeX );
            patch.setPatchSize2DYInPixel( refPatch.getPatchSize2DYInPixel() +
                                          ( mpdu.getMpdu2dDeltaSizeY() ) * quantizerSizeY );

            patch.getSizeU0() =
                ceil( static_cast<double>( patch.getPatchSize2DXInPixel() ) / static_cast<double>( packingBlockSize ) );
            patch.getSizeV0() =
                ceil( static_cast<double>( patch.getPatchSize2DYInPixel() ) / static_cast<double>( packingBlockSize ) );
          } else {
            patch.getSizeU0() = mpdu.getMpdu2dDeltaSizeX() + refPatch.getSizeU0();
            patch.getSizeV0() = mpdu.getMpdu2dDeltaSizeY() + refPatch.getSizeV0();
          }

          if ( asps.getPointLocalReconstructionEnabledFlag() ) { overridePlrFlag = true; }
        } else {
          if ( mpdu.getMpduOverride3dParamsFlag() ) {
            patch.getU1() = mpdu.getMpdu3dPosX() + refPatch.getU1();
            patch.getV1() = mpdu.getMpdu3dPosY() + refPatch.getV1();
            if ( patch.getProjectionMode() == 0 ) {
              patch.getD1() = ( mpdu.getMpdu3dPosMinZ() + ( refPatch.getD1() / minLevel ) ) * minLevel;
            } else {
              if ( static_cast<int>( asps.get45DegreeProjectionPatchPresentFlag() ) == 0 ) {
                patch.getD1() =
                    max3DCoordinate -
                    ( mpdu.getMpdu3dPosMinZ() + ( ( max3DCoordinate - refPatch.getD1() ) / minLevel ) ) * minLevel;
              } else {
                patch.getD1() =
                    ( max3DCoordinate << 1 ) -
                    ( mpdu.getMpdu3dPosMinZ() + ( ( ( max3DCoordinate << 1 ) - refPatch.getD1() ) / minLevel ) ) *
                        minLevel;
              }
            }

            const int64_t delta_DD = mpdu.getMpdu3dPosDeltaMaxZ();
            size_t        prevDD   = refPatch.getSizeD() / minLevel;
            if ( prevDD * minLevel != refPatch.getSizeD() ) { prevDD += 1; }
            patch.getSizeD() = ( std::min )( size_t( ( delta_DD + prevDD ) * minLevel ), (size_t)255 );

            if ( asps.getPointLocalReconstructionEnabledFlag() ) {
              overridePlrFlag = ( mpdu.getMpduOverridePlrFlag() != 0 );
            }
          }
        }
        patch.getProjectionMode()        = refPatch.getProjectionMode();
        patch.getPatchOrientation()      = refPatch.getPatchOrientation();
        patch.getNormalAxis()            = refPatch.getNormalAxis();
        patch.getTangentAxis()           = refPatch.getTangentAxis();
        patch.getBitangentAxis()         = refPatch.getBitangentAxis();
        patch.getAxisOfAdditionalPlane() = refPatch.getAxisOfAdditionalPlane();
        patch.setLodScaleX( refPatch.getLodScaleX() );
        patch.setLodScaleY( refPatch.getLodScaleY() );
        prevSizeU0              = patch.getSizeU0();
        prevSizeV0              = patch.getSizeV0();
        prevPatchSize2DXInPixel = patch.getPatchSize2DXInPixel();
        prevPatchSize2DYInPixel = patch.getPatchSize2DYInPixel();

        TRACE_CODEC(
            "patch(Inter) %zu: UV0 %4zu %4zu UV1 %4zu %4zu D1=%4zu S=%4zu %4zu "
            "%4zu from DeltaSize = "
            "%4ld %4ld P=%zu O=%zu A=%u%u%u Lod = %zu,%zu \n",
            patchIndex, patch.getU0(), patch.getV0(), patch.getU1(), patch.getV1(), patch.getD1(), patch.getSizeU0(),
            patch.getSizeV0(), patch.getSizeD(), mpdu.getMpdu2dDeltaSizeX(), mpdu.getMpdu2dDeltaSizeY(),
            patch.getProjectionMode(), patch.getPatchOrientation(), patch.getNormalAxis(), patch.getTangentAxis(),
            patch.getBitangentAxis(), patch.getLodScaleX(), patch.getLodScaleY() );

        patch.allocOneLayerData();
        if ( asps.getPointLocalReconstructionEnabledFlag() ) {
          setPointLocalReconstructionData( frame, patch, mpdu.getPointLocalReconstructionData(),
                                           context.getOccupancyPackingBlockSize() );
        }
      } else if ( currPatchType == SKIP_PATCH ) {
        assert( -1 );
        auto& patch = patches[patchIndex];
        TRACE_CODEC( "patch %zu / %zu: Inter \n", patchIndex, patchCount );
        TRACE_CODEC( "SDU: refAtlasFrame= 0 refPatchIdx = %d \n", patchIndex );

        patch.setBestMatchIdx( static_cast<int32_t>( patchIndex ) );
        predIndex += patchIndex;
        patch.setRefAtlasFrameIndex( 0 );
        size_t      refPOC   = frame.getRefAFOC( patch.getRefAtlasFrameIndex() );
        const auto& refPatch = context.getFrame( refPOC ).getPatches()[patch.getBestMatchIdx()];
        TRACE_CODEC(
            "\trefPatch: Idx = %zu UV0 = %zu %zu  UV1 = %zu %zu Size = %zu %zu "
            "%zu  Lod = %u,%u\n",
            patch.getBestMatchIdx(), refPatch.getU0(), refPatch.getV0(), refPatch.getU1(), refPatch.getV1(),
            refPatch.getSizeU0(), refPatch.getSizeV0(), refPatch.getSizeD(), refPatch.getLodScaleX(),
            refPatch.getLodScaleY() );

        patch.getProjectionMode()   = refPatch.getProjectionMode();
        patch.getU0()               = refPatch.getU0();
        patch.getV0()               = refPatch.getV0();
        patch.getPatchOrientation() = refPatch.getPatchOrientation();
        patch.getU1()               = refPatch.getU1();
        patch.getV1()               = refPatch.getV1();
        if ( asps.getPatchSizeQuantizerPresentFlag() ) {
          patch.setPatchSize2DXInPixel( refPatch.getPatchSize2DXInPixel() );
          patch.setPatchSize2DYInPixel( refPatch.getPatchSize2DYInPixel() );

          patch.getSizeU0() =
              ceil( static_cast<double>( patch.getPatchSize2DXInPixel() ) / static_cast<double>( packingBlockSize ) );
          patch.getSizeV0() =
              ceil( static_cast<double>( patch.getPatchSize2DYInPixel() ) / static_cast<double>( packingBlockSize ) );
        } else {
          patch.getSizeU0() = refPatch.getSizeU0();
          patch.getSizeV0() = refPatch.getSizeV0();
        }
        patch.getNormalAxis()            = refPatch.getNormalAxis();
        patch.getTangentAxis()           = refPatch.getTangentAxis();
        patch.getBitangentAxis()         = refPatch.getBitangentAxis();
        patch.getAxisOfAdditionalPlane() = refPatch.getAxisOfAdditionalPlane();
        const size_t max3DCoordinate     = size_t( 1 ) << ( gi.getGeometry3dCoordinatesBitdepthMinus1() + 1 );
        if ( patch.getProjectionMode() == 0 ) {
          patch.getD1() = ( ( refPatch.getD1() / minLevel ) ) * minLevel;
        } else {
          if ( static_cast<int>( asps.get45DegreeProjectionPatchPresentFlag() ) == 0 ) {
            patch.getD1() = max3DCoordinate - ( ( ( max3DCoordinate - refPatch.getD1() ) / minLevel ) ) * minLevel;
          } else {
            patch.getD1() = ( max3DCoordinate << 1 ) -
                            ( ( ( ( max3DCoordinate << 1 ) - refPatch.getD1() ) / minLevel ) ) * minLevel;
          }
        }
        size_t prevDD = refPatch.getSizeD() / minLevel;
        if ( prevDD * minLevel != refPatch.getSizeD() ) { prevDD += 1; }
        patch.getSizeD() = ( std::min )( size_t( (prevDD)*minLevel ), (size_t)255 );
        patch.setLodScaleX( refPatch.getLodScaleX() );
        patch.setLodScaleY( refPatch.getLodScaleY() );
        prevSizeU0              = patch.getSizeU0();
        prevSizeV0              = patch.getSizeV0();
        prevPatchSize2DXInPixel = patch.getPatchSize2DXInPixel();
        prevPatchSize2DYInPixel = patch.getPatchSize2DYInPixel();
        TRACE_CODEC(
            "patch(skip) %zu: UV0 %4zu %4zu UV1 %4zu %4zu D1=%4zu S=%4zu %4zu "
            "%4zu P=%zu O=%zu A=%u%u%u Lod = %zu,%zu "
            "\n",
            patchIndex, patch.getU0(), patch.getV0(), patch.getU1(), patch.getV1(), patch.getD1(), patch.getSizeU0(),
            patch.getSizeV0(), patch.getSizeD(), patch.getProjectionMode(), patch.getPatchOrientation(),
            patch.getNormalAxis(), patch.getTangentAxis(), patch.getBitangentAxis(), patch.getLodScaleX(),
            patch.getLodScaleY() );
        patch.allocOneLayerData();
      } else if ( currPatchType == RAW_PATCH ) {
        TRACE_CODEC( "patch %zu / %zu: raw \n", patchIndex, patchCount );
        auto& ppdu             = pid.getRawPatchDataUnit();
        auto& rawPointsPatch   = pcmPatches[patchIndex - numNonRawPatch];
        rawPointsPatch.u0_     = ppdu.getRpdu2dPosX();
        rawPointsPatch.v0_     = ppdu.getRpdu2dPosY();
        rawPointsPatch.sizeU0_ = ppdu.getRpdu2dSizeXMinus1() + 1;
        rawPointsPatch.sizeV0_ = ppdu.getRpdu2dSizeYMinus1() + 1;
        if ( afps.getAfpsRaw3dPosBitCountExplicitModeFlag() ) {
          rawPointsPatch.u1_ = ppdu.getRpdu3dPosX();
          rawPointsPatch.v1_ = ppdu.getRpdu3dPosY();
          rawPointsPatch.d1_ = ppdu.getRpdu3dPosZ();
        } else {
          const size_t pcmU1V1D1Level = size_t( 1 ) << ( gi.getGeometryNominal2dBitdepthMinus1() + 1 );
          rawPointsPatch.u1_          = ppdu.getRpdu3dPosX() * pcmU1V1D1Level;
          rawPointsPatch.v1_          = ppdu.getRpdu3dPosY() * pcmU1V1D1Level;
          rawPointsPatch.d1_          = ppdu.getRpdu3dPosZ() * pcmU1V1D1Level;
        }
        rawPointsPatch.setNumberOfRawPoints( ppdu.getRpduRawPointsMinus1() + 1 );
        rawPointsPatch.occupancyResolution_ = context.getOccupancyPackingBlockSize();
        totalNumberOfRawPoints += rawPointsPatch.getNumberOfRawPoints();
        TRACE_CODEC(
            "Raw :UV = %zu %zu  size = %zu %zu  uvd1 = %zu %zu %zu numPoints = "
            "%zu ocmRes = %zu \n",
            rawPointsPatch.u0_, rawPointsPatch.v0_, rawPointsPatch.sizeU0_, rawPointsPatch.sizeV0_, rawPointsPatch.u1_,
            rawPointsPatch.v1_, rawPointsPatch.d1_, rawPointsPatch.numberOfRawPoints_,
            rawPointsPatch.occupancyResolution_ );
      } else if ( currPatchType == EOM_PATCH ) {
        TRACE_CODEC( "patch %zu / %zu: EOM \n", patchIndex, patchCount );
        auto&       epdu       = pid.getEomPatchDataUnit();
        auto&       eomPatches = frame.getEomPatches();
        PCCEomPatch eomPatch;
        eomPatch.u0_    = epdu.getEpdu2dPosX();
        eomPatch.v0_    = epdu.getEpdu2dPosY();
        eomPatch.sizeU_ = epdu.getEpdu2dSizeXMinus1() + 1;
        eomPatch.sizeV_ = epdu.getEpdu2dSizeYMinus1() + 1;
        eomPatch.memberPatches.resize( epdu.getEpduAssociatedPatchesCountMinus1() + 1 );
        eomPatch.eomCountPerPatch.resize( epdu.getEpduAssociatedPatchesCountMinus1() + 1 );
        eomPatch.eomCount_ = 0;
        for ( size_t i = 0; i < eomPatch.memberPatches.size(); i++ ) {
          eomPatch.memberPatches[i]    = epdu.getEpduAssociatedPatches( i );
          eomPatch.eomCountPerPatch[i] = epdu.getEpduEomPointsPerPatch( i );
          eomPatch.eomCount_ += eomPatch.eomCountPerPatch[i];
        }
        eomPatches.push_back( eomPatch );
        TRACE_CODEC( "EOM: U0V0 %zu,%zu\tSizeU0V0 %zu,%zu\tN= %zu,%zu\n", eomPatch.u0_, eomPatch.v0_, eomPatch.sizeU_,
                     eomPatch.sizeV_, eomPatch.memberPatches.size(), eomPatch.eomCount_ );
        for ( size_t i = 0; i < eomPatch.memberPatches.size(); i++ ) {
          TRACE_CODEC( "%zu, %zu\n", eomPatch.memberPatches[i], eomPatch.eomCountPerPatch[i] );
        }
        TRACE_CODEC( "\n" );
      } else if ( currPatchType == END_PATCH ) {
        break;
      } else {
        printf( "Error: unknow frame/patch type \n" );
        TRACE_CODEC( "Error: unknow frame/patch type \n" );
      }
    }
    TRACE_CODEC( "patch %zu / %zu: end \n", patches.size(), patches.size() );
    frame.setTotalNumberOfRawPoints( totalNumberOfRawPoints );
  }
}
