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
#include "PCCContext.h"
#include "PCCFrameContext.h"
#include "PCCPatch.h"
#include "PCCPatchSegmenter.h"
#include "PCCVideoEncoder.h"
#include "PCCGroupOfFrames.h"
#include "PCCPointSet.h"
#include "PCCEncoderParameters.h"
#include "PCCKdTree.h"
#include <tbb/tbb.h>
#include "PCCChrono.h"
#include "PCCEncoder.h"

uint64_t changedPixCnt;
uint64_t changedPixCnt0To1;
uint64_t changedPixCnt1To0;
uint64_t pixCnt;

using namespace std;
using namespace pcc;

std::string getEncoderConfig1L( const std::string& string ) {
  std::string sub    = string.substr( 0, string.find_last_of( '.' ) );
  std::string result = sub + "-1L.cfg";
  return result;
}

PCCEncoder::PCCEncoder() {
#ifdef ENABLE_PAPI_PROFILING
  initPapiProfiler();
#endif
}

PCCEncoder::~PCCEncoder() = default;

void PCCEncoder::setParameters( const PCCEncoderParameters& params ) { params_ = params; }

int PCCEncoder::encode( const PCCGroupOfFrames& sources, PCCContext& context, PCCGroupOfFrames& reconstructs ) {
  size_t pointLocalReconstructionOriginal     = static_cast<size_t>( params_.pointLocalReconstruction_ );
  size_t layerCountMinus1Original             = params_.mapCountMinus1_;
  size_t singleLayerPixelInterleavingOriginal = static_cast<size_t>( params_.singleMapPixelInterleaving_ );
  if ( params_.nbThread_ > 0 ) { tbb::task_scheduler_init init( static_cast<int>( params_.nbThread_ ) ); }

  if ( sources.getFrameCount() == 0 ) { return 0; }
  assert( sources.getFrameCount() < 256 );
  if ( params_.losslessGeo_ && params_.tileSegmentationType_ > 1 && params_.numMaxTilePerFrame_ > 1 ) {
    params_.numMaxTilePerFrame_ += 1;
  }
  reconstructs.setFrameCount( sources.getFrameCount() );

#if 0  // Note JR: logger examples
  logger_->traceDescr( "test descr = %d \n", 10 );
  logger_->traceDescr( "test descr = %d \n", 11 );
  logger_->traceAtlas( "test atlas = %d \n", 12 );
  logger_->traceAtlas( "test atlas = %d \n", 13 );
  logger_->traceFrame( "test frame = %d \n", 14 );
  logger_->traceFrame( "test frame = %d \n", 15 );
  logger_->traceTiles( "test tiles = %d \n", 16 );
  logger_->traceTiles( "test tiles = %d \n", 17 );
  logger_->traceTrace( "test trace = %d \n", 18 );
  logger_->traceTrace( "test trace = %d \n", 19 );
#endif

  context.resizeAtlas( 1 );
  context.setAtlasIndex( 0 );
  context.resize( sources.getFrameCount() );
  auto& frames = context.getFrames();
  for ( size_t i = 0; i < frames.size(); i++ ) {
    frames[i].getTitleFrameContext().setFrameIndex( i );  // should before setRefAFOCList
    frames[i].getTitleFrameContext().setRawPatchEnabledFlag( params_.losslessGeo_ || params_.lossyRawPointsPatch_ );
    frames[i].getTitleFrameContext().setUseRawPointsSeparateVideo( params_.useRawPointsSeparateVideo_ );
    frames[i].getTitleFrameContext().setGeometry3dCoordinatesBitdepth( params_.geometry3dCoordinatesBitdepth_ +
                                                                       ( params_.additionalProjectionPlaneMode_ > 0 ) );
    frames[i].getTitleFrameContext().setGeometry2dBitdepth( params_.geometryNominal2dBitdepth_ );
    frames[i].getTitleFrameContext().setMaxDepth( ( 1 << params_.geometryNominal2dBitdepth_ ) - 1 );
    frames[i].getTitleFrameContext().setLog2PatchQuantizerSizeX( params_.log2QuantizerSizeX_ );
    frames[i].getTitleFrameContext().setLog2PatchQuantizerSizeY( params_.log2QuantizerSizeY_ );
  }

  // Segmentation
  generateSegments( sources, context );

  // Init context and tiles
  params_.initializeContext( context );

  // Segment Placement
  placeSegments( sources, context );
  // updatePartitionInformation
  if ( params_.tileSegmentationType_ > 1 && params_.numMaxTilePerFrame_ > 1 ) {
    placeTiles( context, params_.minimumImageWidth_, params_.minimumImageHeight_ );
  }
  if ( params_.tileSegmentationType_ > 0 ) { replaceFrameContext( context ); }

  PCCVideoEncoder   videoEncoder;
  size_t            atlasIndex = context.getAtlasIndex();
  const size_t      pointCount = sources[0].getPointCount();
  auto&             sps        = context.getVps();
  auto&             ai         = sps.getAttributeInformation( atlasIndex );
  auto&             asps       = context.getAtlasSequenceParameterSet( atlasIndex );
  std::stringstream path;
  path << removeFileExtension( params_.compressedStreamPath_ ) << "_GOF" << sps.getV3CParameterSetId() << "_";
  sps.setFrameWidth( atlasIndex, static_cast<uint16_t>( frames[0].getAtlasFrameWidth() ) );
  sps.setFrameHeight( atlasIndex, static_cast<uint16_t>( frames[0].getAtlasFrameHeight() ) );
  // DIS requirement, see 7.4.6.1
  for ( int i = 0; i < context.getAtlasSequenceParameterSetList().size(); i++ ) {
    context.getAtlasSequenceParameterSet( i ).setFrameHeight( sps.getFrameHeight( atlasIndex ) );
    context.getAtlasSequenceParameterSet( i ).setFrameWidth( sps.getFrameWidth( atlasIndex ) );
  }

  // GENERATE OCCUPANCY MAP
#if 1
  printf( "Processing Occupancy map\n" );
#endif
  generateOccupancyMap( context, true );

  // ENCODE OCCUPANCY MAP
  auto& videoBitstream = context.createVideoBitstream( VIDEO_OCCUPANCY );
  generateOccupancyMapVideo( sources, context );
  auto& videoOccupancyMap = context.getVideoOccupancyMap();
  videoEncoder.compress( videoOccupancyMap, path.str(), params_.occupancyMapQP_, videoBitstream,
                         params_.occupancyMapVideoEncoderConfig_, params_.videoEncoderOccupancyPath_,
                         params_.videoEncoderOccupancyCodecId_, params_.byteStreamVideoCoderOccupancy_,
                         context,
                         ( params_.EOMFixBitCount_ <= 8 ) ? 1 : 2,  // nByte
                         false,                                     // use444CodecIo
                         false,                                     // use3dmv
                         8,                                         // internalBitDepth
                         false,                                     // useConversion
                         params_.keepIntermediateFiles_ );
  if ( params_.offsetLossyOM_ > 0 ) {
    changedPixCnt     = 0;
    changedPixCnt0To1 = 0;
    changedPixCnt1To0 = 0;
    pixCnt            = 0;
    modifyOccupancyMap( sources, context );
    std::cout << "Percentage of changed occupancy map values = "
              << ( static_cast<float>( changedPixCnt ) * 100.0F / pixCnt ) << std::endl;
    std::cout << "Percentage of changed occupancy map values from 0 to 1 = "
              << ( static_cast<float>( changedPixCnt0To1 ) * 100.0F / pixCnt ) << std::endl;
    std::cout << "Percentage of changed occupancy map values from 1 to 0 = "
              << ( static_cast<float>( changedPixCnt1To0 ) * 100.0F / pixCnt ) << std::endl;
  }
  if ( !params_.useRawPointsSeparateVideo_ && ( params_.losslessGeo_ || params_.lossyRawPointsPatch_ ) ) {
    markRawPatchLocationOccupancyMapVideo( context );
  }

  generateBlockToPatchFromOccupancyMapVideo( context, params_.occupancyResolution_, params_.occupancyPrecision_ );
#if 1
  printf( "Processing Geometry\n" );
  fflush( stdout );
#endif
  // Generate GEOMETRY IMAGE & dilation
  generateGeometryVideo( sources, context );
  printf( "generateGeometryVideo done \n" );
  fflush( stdout );

  // ENCODE GEOMETRY IMAGE
  if ( params_.use3dmc_ ) { create3DMotionEstimationFiles( context, path.str() ); }
  auto&  gi                      = context.getVps().getGeometryInformation( atlasIndex );
  size_t geometryVideoBitDepth   = gi.getGeometry2dBitdepthMinus1() + 1;
  size_t geometryMPVideoBitDepth = gi.getGeometry2dBitdepthMinus1() + 1;
  size_t nbyteGeo                = ( geometryVideoBitDepth <= 8 ) ? 1 : 2;
  size_t nbyteGeoMP              = ( geometryMPVideoBitDepth <= 8 ) ? 1 : 2;
  size_t internalBitDepth        = 10;

  if ( params_.losslessGeo_ ) { internalBitDepth = geometryVideoBitDepth; }

  auto& videoBitstreamD0 = params_.multipleStreams_ ? context.createVideoBitstream( VIDEO_GEOMETRY_D0 )
                                                    : context.createVideoBitstream( VIDEO_GEOMETRY );
  auto&       videoGeometry = context.getVideoGeometryMultiple()[0];
  std::string geometryConfigFile =
      params_.multipleStreams_
          ? params_.geometryD0Config_
          : ( params_.mapCountMinus1_ == 0 ? getEncoderConfig1L( params_.geometryConfig_ ) : params_.geometryConfig_ );
  videoEncoder.compress( videoGeometry,                                                             // video
                         path.str(),                                                                // path
                         params_.multipleStreams_ ? params_.geometryQP_ - 1 : params_.geometryQP_,  // QP
                         videoBitstreamD0,                                                          // bitstream
                         geometryConfigFile,                                                        // config file
                         params_.videoEncoderGeometryPath_,                                         // encoder path
                         params_.videoEncoderGeometryCodecId_,                                      // Codec id
                         params_.byteStreamVideoCoderGeometry_,
                         context,                                                                   // context
                         nbyteGeo,                                                                  // nbyte
                         params_.losslessGeo_ && params_.losslessGeo444_,                           // use444CodecIo
                         params_.use3dmc_,                                                          // use3dmv
                         internalBitDepth,                                                          // internalBitDepth
                         false,                                                                     // useConversion
                         params_.keepIntermediateFiles_ );                                          // keep intermediate
  size_t sizeGeometryVideo = videoBitstreamD0.size();
  std::cout << "sizeGeometryVideo: " << sizeGeometryVideo << std::endl;
  if ( params_.multipleStreams_ ) {
    if ( params_.lossyRawPointsPatch_ ) {
      std::cout << "Error: lossyRawPointsPatch has not been implemented for "
                   "absoluteD1_ = 0 as "
                   "yet. Exiting... "
                << std::endl;
      std::exit( -1 );
    }
    if ( !params_.absoluteD1_ ) {
      // Form differential video geometryD1
      for ( size_t f = 0; f < frames.size(); ++f ) {
        auto& frame1 = context.getVideoGeometryMultiple()[1].getFrame( f );
        predictGeometryFrame( frames[f].getTitleFrameContext(), videoGeometry.getFrame( f ), frame1 );
        dilate3DPadding( sources[f], frames[f], frames[f].getTitleFrameContext(), frame1,
                         videoOccupancyMap.getFrame( f ) );
      }
    }

    // Compress geometryD1
    auto& videoGeometryD1  = context.getVideoGeometryMultiple()[1];
    auto& videoBitstreamD1 = context.createVideoBitstream( VIDEO_GEOMETRY_D1 );
    videoEncoder.compress( videoGeometryD1,                                  // video
                           path.str(),                                       // path
                           params_.geometryQP_,                              // QP
                           videoBitstreamD1,                                 // bitstream
                           params_.geometryD1Config_,                        // config file
                           params_.videoEncoderGeometryPath_,                // encoder path
                           params_.videoEncoderGeometryCodecId_,             // Codec id
                           params_.byteStreamVideoCoderGeometry_,
                           context,                                          // context
                           nbyteGeo,                                         // nbyte
                           params_.losslessGeo_ && params_.losslessGeo444_,  // use444CodecIo
                           params_.use3dmc_,                                 // use3dmv
                           internalBitDepth,                                 // internalBitDepth
                           false,                                            // useConversion
                           params_.keepIntermediateFiles_ );                 // keep intermediate

    size_t sizeGeometryVideoD1 = videoBitstreamD1.size();
    std::cout << "sizeGeometryVideoD1: " << sizeGeometryVideoD1 << std::endl;
    std::cout << "geometryVideo ->" << ( sizeGeometryVideo + sizeGeometryVideoD1 ) << "=" << sizeGeometryVideo << "+"
              << sizeGeometryVideoD1 << " B ("
              << ( ( sizeGeometryVideo + sizeGeometryVideoD1 ) * 8.0 ) / ( 2 * frames.size() * pointCount ) << " bpp)"
              << std::endl;
  }

  if ( asps.getRawPatchEnabledFlag() && asps.getAuxiliaryVideoEnabledFlag() ) {
    std::cout << "*******Video: Aux (Geometry) ********" << std::endl;
    placeAuxiliaryPointsTiles( context );
    auto& videoBitstreamMP = context.createVideoBitstream( VIDEO_GEOMETRY_RAW );
    generateRawPointsGeometryVideo( context );
    auto& videoRawPointsGeometry = context.getVideoRawPointsGeometry();
    videoEncoder.compress( videoRawPointsGeometry, path.str(),
                           params_.lossyRawPointsPatch_ ? params_.lossyRawPointPatchGeoQP_ : params_.geometryQP_,
                           videoBitstreamMP, params_.geometryAuxVideoConfig_, params_.videoEncoderGeometryPath_,
                           params_.videoEncoderGeometryCodecId_, params_.byteStreamVideoCoderGeometry_,
                           context,
                           nbyteGeoMP,        // nbyte
                           false,             // use444CodecIo
                           false,             // use3dmv
                           internalBitDepth,  // internalBitDepth
                           false,             // useConversion
                           params_.keepIntermediateFiles_ );

    if ( params_.lossyRawPointsPatch_ ) {
      for ( size_t fi = 0; fi < context.size(); fi++ ) generateRawPointsGeometryfromVideo( context, fi );
    }
  }

#if 1
  printf( "****TileInfo***Summary******************\n" );
  fflush( stdout );
  for ( size_t fi = 0; fi < context.size(); fi++ ) {
    for ( size_t ti = 0; ti < context[fi].getNumTilesInAtlasFrame(); ti++ ) {
      auto& tile = context[fi].getTile( ti );
      printf(
          "frame %zu tile %zu: (%zu,%zu) (%zux%zu) patchCount:%zu regPoints:%zu, rawPatch:%zu rawPoints:%zu "
          "eomPatch:%zu eomPoints:%zu\n",
          fi, ti, tile.getLeftTopXInFrame(), tile.getLeftTopYInFrame(), tile.getWidth(), tile.getHeight(),
          tile.getPatches().size(), tile.getTotalNumberOfRegularPoints(), tile.getRawPointsPatches().size(),
          tile.getRawPointsPatches().size() == 0 ? 0 : tile.getRawPointsPatch( 0 ).getNumberOfRawPoints(),
          tile.getEomPatches().size(), tile.getEomPatches().size() == 0 ? 0 : tile.getEomPatches( 0 ).eomCount_ );
      if ( params_.useRawPointsSeparateVideo_ ) {
        printf( "frame(auxVideo) %zu tile %zu: (0,%zu) (%zux%zu)\n", fi, ti, context.getAuxTileLeftTopY( ti ),
                context.getAuxVideoWidth(), context.getAuxTileHeight( ti ) );
      }
    }
  }
  printf( "*********************************************\n" );
  fflush( stdout );
#endif
  // RECONSTRUCT POINT CLOUD GEOMETRY
  GeneratePointCloudParameters gpcParams;
  setGeneratePointCloudParameters( gpcParams, context );
  context.allocOneLayerData();
  std::vector<std::vector<uint32_t>> partitions;
  partitions.resize( context.size() );
  for ( size_t frameIdx = 0; frameIdx < context.size(); frameIdx++ ) {
    auto& frame = context[frameIdx];
    for ( size_t tileIdx = 0; tileIdx < frame.getNumTilesInAtlasFrame(); tileIdx++ ) {
      PCCPointSet3 tileReconstrct;
      auto&        tile = frame.getTile( tileIdx );
      if ( params_.pointLocalReconstruction_ ) {
        auto& videoGeometryMultiple = context.getVideoGeometryMultiple();
        pointLocalReconstructionSearch( context, tile, videoGeometryMultiple, gpcParams );
      }
      generatePointCloud( tileReconstrct, context, frameIdx, tileIdx, gpcParams, partitions[frameIdx], false );
      reconstructs[frameIdx].appendPointSet( tileReconstrct );
      if ( frame.getNumTilesInAtlasFrame() != 1 ) {
        frame.getTitleFrameContext().appendPointToPixel( tile.getPointToPixel() );
      }
    }
  }

  if ( ai.getAttributeCount() > 0 ) {
    std::cout << "Texture Coding starts" << std::endl;
    const size_t mapCount = params_.mapCountMinus1_ + 1;
    // GENERATE ATTRIBUTE
    printf( "generateTextureVideo \n" );
    fflush( stdout );
    generateTextureVideo( sources, reconstructs, context, params_ );

    printf( "generateTextureVideo done \n" );
    fflush( stdout );
    if ( !( params_.losslessGeo_ && params_.textureDilationOffLossless_ ) && params_.textureBGFill_ < 3 ) {
      // ATTRIBUTE IMAGE PADDING
      tbb::task_arena limited( static_cast<int>( params_.nbThread_ ) );
      limited.execute( [&] {
        tbb::parallel_for( size_t( 0 ), frames.size(), [&]( const size_t f ) {
          // for ( size_t f = 0; f < frames.size(); ++f ) {
          using namespace std::chrono;
          pcc::chrono::Stopwatch<std::chrono::steady_clock> clockPadding;
          clockPadding.start();
          if ( params_.absoluteT1_ ) {
            switch ( params_.textureBGFill_ ) {
              case 0:
                for ( int mapIdx = 0; mapIdx < mapCount; mapIdx++ ) {
                  size_t videoFrameIdx = params_.multipleStreams_ ? f : ( f * mapCount + mapIdx );
                  auto&  videoTexture  = context.getVideoTextureMultiple()[params_.multipleStreams_ ? mapIdx : 0];
                  dilate( frames[f].getTitleFrameContext(), videoTexture.getFrame( videoFrameIdx ) );
                }
                break;
              case 1:
                for ( int mapIdx = 0; mapIdx < mapCount; mapIdx++ ) {
                  size_t videoFrameIdx = params_.multipleStreams_ ? f : ( f * mapCount + mapIdx );
                  auto&  videoTexture  = context.getVideoTextureMultiple()[params_.multipleStreams_ ? mapIdx : 0];
                  dilateSmoothedPushPull( frames[f].getTitleFrameContext(), videoTexture.getFrame( videoFrameIdx ) );
                }
                break;
              case 2:
                for ( int mapIdx = 0; mapIdx < mapCount; mapIdx++ ) {
                  size_t videoFrameIdx = params_.multipleStreams_ ? f : ( f * mapCount + mapIdx );
                  auto&  videoTexture  = context.getVideoTextureMultiple()[params_.multipleStreams_ ? mapIdx : 0];
                  dilateHarmonicBackgroundFill( frames[f].getTitleFrameContext(),
                                                videoTexture.getFrame( videoFrameIdx ) );
                }
                break;
              default: std::cout << "Warning: no texture padding applied!" << std::endl;
            }  // switch
            if ( mapCount > 1 && !params_.multipleStreams_ && params_.groupDilation_ ) {
              // Group dilation in texture
              auto& frame        = frames[f].getTitleFrameContext();
              auto& occupancyMap = frame.getOccupancyMap();
              auto& width        = frame.getWidth();
              auto& height       = frame.getHeight();
              auto& frame1       = context.getVideoTextureMultiple()[0].getFrame( f * mapCount );
              auto& frame2       = context.getVideoTextureMultiple()[0].getFrame( f * mapCount + 1 );
              for ( size_t y = 0; y < height; y++ ) {
                for ( size_t x = 0; x < width; x++ ) {
                  const size_t pos = y * width + x;
                  if ( occupancyMap[pos] == 0 ) {
                    for ( size_t c = 0; c < 3; c++ ) {
                      uint8_t  tmp_d0  = frame1.getValue( c, x, y );
                      uint8_t  tmp_d1  = frame2.getValue( c, x, y );
                      uint32_t tmp_avg = ( static_cast<uint32_t>( tmp_d0 ) + static_cast<uint32_t>( tmp_d1 ) + 1 ) >> 1;
                      frame1.setValue( c, x, y, static_cast<uint8_t>( tmp_avg ) );
                      frame2.setValue( c, x, y, static_cast<uint8_t>( tmp_avg ) );
                    }
                  }
                }
              }
            }  // groupDilation and !onelayerMode
          }    // absoluteT1
          else if ( params_.multipleStreams_ ) {
            // params_.multipleStreams_ && !absoluteT1
            auto& frame = context.getVideoTextureMultiple()[0].getFrame( f );
            switch ( params_.textureBGFill_ ) {
              case 0: dilate( frames[f].getTitleFrameContext(), frame ); break;
              case 1: dilateSmoothedPushPull( frames[f].getTitleFrameContext(), frame ); break;
              case 2: dilateHarmonicBackgroundFill( frames[f].getTitleFrameContext(), frame ); break;
              default: std::cout << "Warning: no texture padding applied!" << std::endl;
            }
          }
          clockPadding.stop();
          using ms              = milliseconds;
          auto totalPaddingTime = duration_cast<ms>( clockPadding.count() ).count();
          std::cout << "Processing time (Padding [T0T1]" << f << "/" << frames.size()
                    << "): " << totalPaddingTime / 1000.0 << " s\n";
        } );
      } );
      // }
    }
    // ENCODE ATTRIBUTE IMAGE
    std::cout << "texture video " << std::endl;
    auto& videoBitstream = params_.multipleStreams_ ? context.createVideoBitstream( VIDEO_TEXTURE_T0 )
                                                    : context.createVideoBitstream( VIDEO_TEXTURE );
    const size_t nbyteAtt = 1;
    videoEncoder.compress(
        context.getVideoTextureMultiple()[0], path.str(), params_.textureQP_, videoBitstream,
        params_.multipleStreams_
            ? ( params_.mapCountMinus1_ == 0 ? getEncoderConfig1L( params_.textureConfig_ ) : params_.textureT0Config_ )
            : ( params_.mapCountMinus1_ == 0 ? getEncoderConfig1L( params_.textureConfig_ ) : params_.textureConfig_ ),
        params_.videoEncoderAttributePath_, params_.videoEncoderAttributeCodecId_, params_.byteStreamVideoCoderAttribute_, context, nbyteAtt,  // nbyte
        params_.losslessGeo_,                                                                          // use444CodecIo
        params_.use3dmc_,                                                                              // use3dmv
        params_.losslessGeo_ ? 8 : 10,               // internalBitDepth
        !params_.losslessGeo_,                       // useConversion
        params_.keepIntermediateFiles_,              // keepIntermediateFiles
        params_.colorSpaceConversionConfig_,         // colorSpaceConversionConfig
        params_.inverseColorSpaceConversionConfig_,  // inverseColorSpaceConversionConfig
        params_.colorSpaceConversionPath_ );         // colorSpaceConversionPath
    auto sizeTextureVideo = videoBitstream.size();
    std::cout << "texture video ->" << sizeTextureVideo << " B ("
              << ( sizeTextureVideo * 8.0 ) / ( 2 * frames.size() * pointCount ) << " bpp)" << std::endl;

    if ( params_.multipleStreams_ ) {
      // Form differential video textureT1
      if ( !params_.absoluteT1_ ) {
        for ( size_t f = 0; f < frames.size(); ++f ) {
          auto& frame0 = context.getVideoTextureMultiple()[0].getFrame( f );
          auto& frame1 = context.getVideoTextureMultiple()[1].getFrame( f );
          predictTextureFrame( frames[f].getTitleFrameContext(), frame0, frame1 );
          if ( !( params_.losslessGeo_ && params_.textureDilationOffLossless_ ) ) {
            switch ( params_.textureBGFill_ ) {
              case 0: dilate( frames[f].getTitleFrameContext(), frame1 ); break;
              case 1: dilateSmoothedPushPull( frames[f].getTitleFrameContext(), frame1 ); break;
              case 2: dilateHarmonicBackgroundFill( frames[f].getTitleFrameContext(), frame1 ); break;
              default: std::cout << "Warning: no texture padding applied!" << std::endl;
            }
          }
        }
        std::cout << "texture prediction done " << std::endl;
      }  //! absoluteT1

      // compress textureT1
      auto& videoBitstreamT1 = context.createVideoBitstream( VIDEO_TEXTURE_T1 );
      videoEncoder.compress(
          context.getVideoTextureMultiple()[1], path.str(), params_.textureQP_ + params_.qpAdjT1_, videoBitstreamT1,
          params_.mapCountMinus1_ == 0 ? getEncoderConfig1L( params_.textureConfig_ ) : params_.textureT1Config_,
          params_.videoEncoderAttributePath_, params_.videoEncoderAttributeCodecId_, params_.byteStreamVideoCoderAttribute_, context, nbyteAtt,  // nbyte
          params_.losslessGeo_,                        // use444CodecIo
          params_.use3dmc_,                            // use3dmv
          params_.losslessGeo_ ? 8 : 10,               // internalBitDepth
          !params_.losslessGeo_,                       // useConversion
          params_.keepIntermediateFiles_,              // keepIntermediateFiles
          params_.colorSpaceConversionConfig_,         // colorSpaceConversionConfig
          params_.inverseColorSpaceConversionConfig_,  // inverseColorSpaceConversionConfig
          params_.colorSpaceConversionPath_ );

      size_t sizeTextureVideoT1 = videoBitstreamT1.size();
      std::cout << "texture video ->" << ( sizeTextureVideo + sizeTextureVideoT1 ) << "=" << sizeTextureVideo << "+"
                << sizeTextureVideoT1 << " B ("
                << ( ( sizeTextureVideo + sizeTextureVideoT1 ) * 8.0 ) / ( 2 * frames.size() * pointCount ) << " bpp)"
                << std::endl;
    }

    if ( asps.getRawPatchEnabledFlag() && asps.getAuxiliaryVideoEnabledFlag() ) {
      std::cout << "*******Video: Aux (Texture) ********" << std::endl;
      auto& videoBitstreamMP = context.createVideoBitstream( VIDEO_TEXTURE_RAW );
      generateRawPointsTextureVideo( context );
      auto&        videoRawPointsTexture = context.getVideoRawPointsTexture();
      const size_t nByteAttMP            = 1;
      videoEncoder.compress( videoRawPointsTexture, path.str(), params_.textureQP_, videoBitstreamMP,
                             params_.textureAuxVideoConfig_, params_.videoEncoderAttributePath_,
                             params_.videoEncoderAttributeCodecId_, params_.byteStreamVideoCoderAttribute_, context,
                             nByteAttMP,                                  // nbyte
                             params_.losslessGeo_,                        // use444CodecIo
                             false,                                       // use3dmv
                             10,                                          // internalBitDepth
                             !params_.losslessGeo_,                       // useConversion
                             params_.keepIntermediateFiles_,              // keepIntermediateFiles
                             params_.colorSpaceConversionConfig_,         // colorSpaceConversionConfig
                             params_.inverseColorSpaceConversionConfig_,  // inverseColorSpaceConversionConfig
                             params_.colorSpaceConversionPath_ );         // colorSpaceConversionPath
      if ( params_.lossyRawPointsPatch_ ) {
        printf( "generateRawPointsTexturefromVideo \n" );
        for ( size_t fi = 0; fi < context.size(); fi++ ) generateRawPointsTexturefromVideo( context, fi );
      }
    }
  }

  if ( params_.flagGeometrySmoothing_ ) {
    if ( params_.pbfEnableFlag_ ) {
      gpcParams.pbfEnableFlag_    = true;
      gpcParams.pbfFilterSize_    = params_.pbfFilterSize_;
      gpcParams.pbfPassesCount_   = params_.pbfPassesCount_;
      gpcParams.pbfLog2Threshold_ = params_.pbfLog2Threshold_;
      for ( auto& reconstruct : reconstructs ) { reconstruct.clear(); }
      for ( auto& partition : partitions ) { partition.clear(); }
      partitions.clear();
      partitions.resize( context.size() );
      for ( size_t fi = 0; fi < context.size(); fi++ ) {
        for ( size_t ti = 0; ti < context[fi].getNumTilesInAtlasFrame(); ti++ ) {
          generatePointCloud( reconstructs[fi], context, fi, ti, gpcParams, partitions[fi], false );
        }
      }
    }
  }
  std::cout << "Color Point Clouds" << std::endl;
  // RECOLOR RECONSTRUCTED POINT CLOUD
  // recreating the prediction list per attribute (either the attribute is coded
  // absoulte, or follows the geometry)
  // see contribution m52529

  std::vector<std::vector<bool>> absoluteT1List;
  absoluteT1List.resize( ai.getAttributeCount() );
  for ( int attrIdx = 0; attrIdx < ai.getAttributeCount(); ++attrIdx ) {
    for ( uint32_t mapIdx = 0; mapIdx < sps.getMapCountMinus1( atlasIndex ) + 1; ++mapIdx ) {
      absoluteT1List[attrIdx].push_back( params_.absoluteT1_ );
    }
  }
  for ( size_t frameIdx = 0; frameIdx < context.size(); frameIdx++ ) {
    reconstructs[frameIdx].addColors();
    reconstructs[frameIdx].addColors16bit();

    size_t accTilePointCount = 0;
    for ( size_t tileIdx = 0; tileIdx < context[frameIdx].getNumTilesInAtlasFrame(); tileIdx++ ) {
      auto& tile = context[frameIdx].getTile( tileIdx );
      for ( size_t attIdx = 0; attIdx < 1; attIdx++ ) {  // ai.getAttributeCount()
        size_t updatedPointCount = colorPointCloud( reconstructs[frameIdx], context, tile, absoluteT1List[attIdx],
                                                    sps.getMultipleMapStreamsPresentFlag( 0 ), ai.getAttributeCount(),
                                                    accTilePointCount, gpcParams );
        accTilePointCount        = updatedPointCount;
      }
    }  // tile
  }

  std::cout << "Post Processing Point Clouds" << std::endl;
  bool isAttributes444 = static_cast<int>( params_.losslessGeo_ ) == 1;
  for ( size_t frameIdx = 0; frameIdx < sources.getFrameCount(); frameIdx++ ) {
    auto&                        frame = context.getFrame( frameIdx );
    GeneratePointCloudParameters ppSEIParams;
    setPostProcessingSeiParameters( ppSEIParams, context );
    auto& reconstruct = reconstructs[frameIdx];
    auto& partition   = partitions[frameIdx];

    TRACE_PATCH( "Post-Processing: postprocessSmoothing = %zu pbfEnableFlag = %d \n",
                 params_.postprocessSmoothingFilter_, params_.pbfEnableFlag_ );
    if ( ppSEIParams.flagGeometrySmoothing_ ) {
      PCCPointSet3 tempFrameBuffer = reconstruct;
      if ( ppSEIParams.gridSmoothing_ ) {
        smoothPointCloudPostprocess( reconstruct, params_.colorTransform_, ppSEIParams, partition );
      }
      if ( !ppSEIParams.pbfEnableFlag_ ) {
        // These are different attribute transfer functions
        if ( params_.postprocessSmoothingFilter_ == 1 || params_.postprocessSmoothingFilter_ == 5 ) {
          TRACE_PATCH( " transferColors16bitBP \n" );
          // tempFrameBuffer[i].transferColors16bit( reconstructs[i], int32_t( 0
          // ), params_.losslessGeo_ == 1, 8, 1, 1,
          // 1, 1, 0, 4, 4, 1000, 1000, 1000 * 256, 1000 * 256 );
          tempFrameBuffer.transferColors16bitBP( reconstruct, params_.postprocessSmoothingFilter_, int32_t( 0 ),
                                                 (bool)( params_.losslessGeo_ ), 8, 1, true, true, true, false, 4, 4,
                                                 1000, 1000, 1000 * 256, 1000 * 256 );
        } else if ( params_.postprocessSmoothingFilter_ == 2 ) {
          TRACE_PATCH( " transferColorWeight \n" );
          tempFrameBuffer.transferColorWeight( reconstruct, 0.1 );
        } else if ( params_.postprocessSmoothingFilter_ == 3 ) {
          TRACE_PATCH( " transferColorsFilter3 \n" );
          tempFrameBuffer.transferColorsFilter3( reconstruct, int32_t( 0 ), isAttributes444 );
        } else if ( params_.postprocessSmoothingFilter_ == 7 || params_.postprocessSmoothingFilter_ == 9 ) {
          TRACE_PATCH( " transferColorsFilter3 \n" );
          tempFrameBuffer.transferColorsBackward16bitBP( reconstruct, params_.postprocessSmoothingFilter_, int32_t( 0 ),
                                                         isAttributes444, 8, 1, true, true, true, false, 4, 4, 1000,
                                                         1000, 1000 * 256, 1000 * 256 );
        }
      }
    }

    if ( ppSEIParams.flagColorSmoothing_ ) {
      TRACE_PATCH( " colorSmoothing \n" );
      colorSmoothing( reconstruct, params_.colorTransform_, ppSEIParams );
    }
    if ( !isAttributes444 ) {  // lossy: convert 16-bit yuv444 to 8-bit RGB444
      TRACE_PATCH( "lossy: convert 16-bit yuv444 to 8-bit RGB444 (convertYUV16ToRGB8) \n" );
      reconstruct.convertYUV16ToRGB8();
      // #ifdef TRACE_CODEC
      //       for ( size_t i = 0; i < 100; i++ ) {
      //         TRACE_PATCH( "%4zu %4zu %4zu: c16 %4zu %4zu %4zu c8 %4zu %4zu %4zu\n", reconstruct[i][0],
      //         reconstruct[i][1],
      //                      reconstruct[i][2], reconstruct.getColor16bit( i )[0], reconstruct.getColor16bit( i )[1],
      //                      reconstruct.getColor16bit( i )[2], reconstruct.getColor( i )[0], reconstruct.getColor( i
      //                      )[1], reconstruct.getColor( i )[2] );
      //       }
      // #endif
    } else {  // lossless: copy 16-bit RGB to 8-bit RGB
      TRACE_PATCH( "lossy: lossless: copy 16-bit RGB to 8-bit RGB (copyRGB16ToRGB8) \n" );
      reconstruct.copyRGB16ToRGB8();
    }
  }  // frame

  if ( !params_.keepIntermediateFiles_ && params_.use3dmc_ ) { remove3DMotionEstimationFiles( path.str() ); }

  createPatchFrameDataStructure( context );
  params_.pointLocalReconstruction_   = ( pointLocalReconstructionOriginal != 0u );
  params_.mapCountMinus1_             = layerCountMinus1Original;
  params_.singleMapPixelInterleaving_ = ( singleLayerPixelInterleavingOriginal != 0u );
  printf( "Done Encoder \n" );
  fflush( stdout );
  return 0;
}

void PCCEncoder::printMap( std::vector<bool> img, const size_t sizeU, const size_t sizeV ) {
  std::cout << std::endl;
  std::cout << "PrintMap size = " << sizeU << " x " << sizeV << std::endl;
  for ( size_t v = 0; v < sizeV; ++v ) {
    for ( size_t u = 0; u < sizeU; ++u ) { std::cout << ( img[v * sizeU + u] ? 'X' : '.' ); }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

void PCCEncoder::printMapTetris( std::vector<bool> img,
                                 const size_t      sizeU,
                                 const size_t      sizeV,
                                 std::vector<int>  horizon ) {
  std::cout << std::endl;
  std::cout << "PrintMap size = " << sizeU << " x " << sizeV << std::endl;
  for ( int v = 0; v < sizeV; ++v ) {
    for ( int u = 0; u < sizeU; ++u ) {
      if ( v == horizon[u] ) {
        std::cout << ( img[v * sizeU + u] ? 'U' : 'O' );
      } else {
        std::cout << ( img[v * sizeU + u] ? 'X' : '.' );
      }
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

static const std::vector<int32_t> kernel = {12, 28, 12, 28, 96, 28, 12, 28, 12};

template <typename T>
T PCCEncoder::limit( T x, T minVal, T maxVal ) {
  return ( x < minVal ) ? minVal : ( x > maxVal ? maxVal : x );
}

void PCCEncoder::preFilterOccupancyMap( PCCImageOccupancyMap& image, size_t kwidth, size_t kheight ) {
  if ( kwidth == 0 ) { kwidth = sqrt( kernel.size() ); }
  if ( kheight == 0 ) { kheight = sqrt( kernel.size() ); }

  const size_t width  = image.getWidth();
  const size_t height = image.getHeight();

  const size_t kCenterW = kwidth / 2;
  const size_t kCenterH = kheight / 2;

  const auto imageTemp( image );
  int        val;
  for ( size_t v = 0; v < height; v++ ) {
    for ( size_t u = 0; u < width; u++ ) {
      val = 0;
      for ( size_t n = 0; n < kheight; n++ ) {
        size_t nn = kheight - 1 - n;
        for ( size_t m = 0; m < kwidth; m++ ) {
          size_t mm = kwidth - 1 - m;
          size_t q  = nn * kwidth + mm;

          size_t i = limit<int32_t>( int32_t( u + kCenterW - mm ), 0, width - 1 );
          size_t j = limit<int32_t>( int32_t( v + kCenterH - nn ), 0, height - 1 );
          val += static_cast<double>( imageTemp.getValue( 0, i, j ) ) * kernel[q];
        }
      }
      image.setValue( 0, u, v, static_cast<uint8_t>( val >> 8 ) );
    }
  }
}

bool PCCEncoder::generateOccupancyMapVideo( const PCCGroupOfFrames& sources, PCCContext& context ) {
  auto& videoOccupancyMap = context.getVideoOccupancyMap();
  bool  ret               = true;
  videoOccupancyMap.resize( sources.getFrameCount() );
  for ( size_t f = 0; f < sources.getFrameCount(); ++f ) {
    auto&                 contextFrame = context.getFrames()[f];
    PCCImageOccupancyMap& videoFrame   = videoOccupancyMap.getFrame( f );
    ret &= generateOccupancyMapVideo( contextFrame.getAtlasFrameWidth(), contextFrame.getAtlasFrameHeight(),
                                      contextFrame.getTitleFrameContext().getOccupancyMap(), videoFrame );
  }
  return ret;
}

bool PCCEncoder::generateOccupancyMapVideo( const size_t           imageWidth,
                                            const size_t           imageHeight,
                                            std::vector<uint32_t>& occupancyMap,
                                            PCCImageOccupancyMap&  videoFrameOccupancyMap ) {
  const size_t   blockSize0  = params_.occupancyResolution_ / params_.occupancyPrecision_;
  const size_t   pointCount0 = blockSize0 * blockSize0;
  vector<bool>   block0;
  vector<size_t> bestRuns;
  vector<size_t> runs;
  block0.resize( pointCount0 );
  size_t       videoFrameOccupancyMapSizeU = imageWidth / params_.occupancyPrecision_;
  size_t       videoFrameOccupancyMapSizeV = imageHeight / params_.occupancyPrecision_;
  const size_t blockToPatchWidth           = imageWidth / params_.occupancyResolution_;
  const size_t blockToPatchHeight          = imageHeight / params_.occupancyResolution_;

  if ( !params_.enhancedOccupancyMapCode_ ) {
    videoFrameOccupancyMap.resize( videoFrameOccupancyMapSizeU, videoFrameOccupancyMapSizeV, PCCCOLORFORMAT::YUV420 );
    for ( size_t v0 = 0; v0 < blockToPatchHeight; ++v0 ) {
      for ( size_t u0 = 0; u0 < blockToPatchWidth; ++u0 ) {
        size_t fullCount = 0;
        for ( size_t v1 = 0; v1 < blockSize0; ++v1 ) {
          const size_t v2 = v0 * params_.occupancyResolution_ + v1 * params_.occupancyPrecision_;
          for ( size_t u1 = 0; u1 < blockSize0; ++u1 ) {
            const size_t u2     = u0 * params_.occupancyResolution_ + u1 * params_.occupancyPrecision_;
            bool         isFull = false;
            for ( size_t v3 = 0; v3 < params_.occupancyPrecision_ && !isFull; ++v3 ) {
              for ( size_t u3 = 0; u3 < params_.occupancyPrecision_ && !isFull; ++u3 ) {
                isFull |= ( occupancyMap[( v2 + v3 ) * imageWidth + u2 + u3] > 0 );
              }
            }
            block0[v1 * blockSize0 + u1] = isFull;
            fullCount += static_cast<unsigned long long>( isFull );
            /*for ( size_t v3 = 0; v3 < params_.occupancyPrecision_; ++v3 ) {
              for ( size_t u3 = 0; u3 < params_.occupancyPrecision_; ++u3 ) {
                occupancyMap[( v2 + v3 ) * imageWidth + u2 + u3] = isFull; ->
            does not update the occupancy map, this
            will be done somewhere else
            }
            }*/
          }
        }
        for ( size_t iterBlockV = 0; iterBlockV < blockSize0; iterBlockV++ ) {
          for ( size_t iterBlockU = 0; iterBlockU < blockSize0; iterBlockU++ ) {
            uint8_t pixel = static_cast<uint8_t>( block0[iterBlockV * blockSize0 + iterBlockU] );
            if ( pixel > 0 ) { pixel = ( params_.offsetLossyOM_ > 0 ) ? params_.offsetLossyOM_ : 1; }
            size_t videoFrameU = u0 * blockSize0 + iterBlockU;
            size_t videoFrameV = v0 * blockSize0 + iterBlockV;
            videoFrameOccupancyMap.setValue( 0, videoFrameU, videoFrameV, pixel );
          }
        }
      }
    }
  } else {
    videoFrameOccupancyMap.resize( imageWidth, imageHeight, PCCCOLORFORMAT::YUV420 );
    for ( size_t v = 0; v < imageHeight; v++ ) {
      for ( size_t u = 0; u < imageWidth; u++ ) {
        size_t i      = v * imageWidth + u;
        size_t symbol = occupancyMap[i];
        if ( symbol < 0 ) { symbol = 0; }
        if ( symbol > 1023 ) { symbol = 1023; }
        videoFrameOccupancyMap.setValue( 0, u, v, symbol );
      }
    }
  }

  if ( params_.prefilterLossyOM_ ) { preFilterOccupancyMap( videoFrameOccupancyMap, 3, 3 ); }

  return true;
}

bool PCCEncoder::modifyOccupancyMap( const PCCGroupOfFrames& sources, PCCContext& context ) {
  std::ofstream oFile;
  if ( params_.keepIntermediateFiles_ ) { oFile.open( "occupancyMap.rgb", std::ios::binary ); }
  auto& videoOccupancyMap = context.getVideoOccupancyMap();
  bool  ret               = true;
  for ( size_t f = 0; f < sources.getFrameCount(); ++f ) {
    auto&                 contextFrame = context.getFrames()[f].getTitleFrameContext();
    PCCImageOccupancyMap& videoFrame   = videoOccupancyMap.getFrame( f );
    ret &= modifyOccupancyMap( contextFrame.getWidth(), contextFrame.getHeight(), contextFrame.getOccupancyMap(),
                               videoFrame, oFile );
  }
  if ( params_.keepIntermediateFiles_ ) { oFile.close(); }
  return ret;
}

bool PCCEncoder::modifyOccupancyMap( const size_t           imageWidth,
                                     const size_t           imageHeight,
                                     std::vector<uint32_t>& occupancyMap,
                                     PCCImageOccupancyMap&  videoFrameOccupancyMap,
                                     std::ofstream&         ofile ) {
  const size_t numSubBlksV = imageHeight / params_.occupancyPrecision_;
  const size_t numSubBlksH = imageWidth / params_.occupancyPrecision_;

  // const size_t threshold = OM_OFFSET / 2;

  std::vector<uint32_t> newOccupancyMap;
  newOccupancyMap.resize( imageWidth * imageHeight );
  char tmpC;

  for ( size_t v0 = 0; v0 < numSubBlksV; ++v0 ) {
    const size_t v1 = v0 * params_.occupancyPrecision_;
    for ( size_t u0 = 0; u0 < numSubBlksH; ++u0 ) {
      const size_t u1    = u0 * params_.occupancyPrecision_;
      uint8_t      pixel = videoFrameOccupancyMap.getValue( 0, u0, v0 );
      for ( size_t v2 = 0; v2 < params_.occupancyPrecision_; v2++ ) {
        for ( size_t u2 = 0; u2 < params_.occupancyPrecision_; u2++ ) {
          size_t index = ( v1 + v2 ) * imageWidth + u1 + u2;
          pixCnt++;
          if ( pixel <= params_.thresholdLossyOM_ ) {
            newOccupancyMap[index] = 0;
          } else {
            newOccupancyMap[index] = 1;
          }

          if ( occupancyMap[index] != newOccupancyMap[index] ) {
            changedPixCnt++;
            if ( occupancyMap[index] == 0 ) {
              changedPixCnt0To1++;
              if ( params_.keepIntermediateFiles_ ) {
                tmpC = static_cast<char>( 255 );
                ofile.write( &tmpC, 1 );
                tmpC = static_cast<char>( 0 );
                ofile.write( &tmpC, 1 );
                tmpC = static_cast<char>( 0 );
                ofile.write( &tmpC, 1 );
              }
            } else {
              changedPixCnt1To0++;
              if ( params_.keepIntermediateFiles_ ) {
                tmpC = static_cast<char>( 0 );
                ofile.write( &tmpC, 1 );
                tmpC = static_cast<char>( 255 );
                ofile.write( &tmpC, 1 );
                tmpC = static_cast<char>( 0 );
                ofile.write( &tmpC, 1 );
              }
            }
          } else {
            if ( occupancyMap[index] == 0 ) {
              if ( params_.keepIntermediateFiles_ ) {
                tmpC = static_cast<char>( 0 );
                ofile.write( &tmpC, 1 );
                tmpC = static_cast<char>( 0 );
                ofile.write( &tmpC, 1 );
                tmpC = static_cast<char>( 0 );
                ofile.write( &tmpC, 1 );
              }
            } else {
              if ( params_.keepIntermediateFiles_ ) {
                tmpC = static_cast<char>( 255 );
                ofile.write( &tmpC, 1 );
                tmpC = static_cast<char>( 255 );
                ofile.write( &tmpC, 1 );
                tmpC = static_cast<char>( 255 );
                ofile.write( &tmpC, 1 );
              }
            }
          }
          occupancyMap[index] = newOccupancyMap[index];
        }
      }
    }
  }

  for ( size_t yy = 0; yy < videoFrameOccupancyMap.getHeight(); yy++ )
    for ( size_t xx = 0; xx < videoFrameOccupancyMap.getWidth(); xx++ ) {
      auto pixel = videoFrameOccupancyMap.getValue( 0, xx, yy );
      videoFrameOccupancyMap.setValue( 0, xx, yy, ( pixel <= params_.thresholdLossyOM_ ) ? 0 : 1 );
    }

  return true;
}

void PCCEncoder::modifyOccupancyMapEOM( PCCFrameContext& tile ) {
  auto& occupancyMap     = tile.getOccupancyMap();
  auto& fullOccupancyMap = tile.getFullOccupancyMap();
  auto& width            = tile.getWidth();
  auto& height           = tile.getHeight();
  occupancyMap.resize( width * height, 0 );
  if ( !params_.absoluteD1_ || !params_.absoluteT1_ ) { fullOccupancyMap.resize( width * height, 0 ); }
  for ( auto& patch : tile.getPatches() ) {
    for ( size_t v = 0; v < patch.getSizeV(); ++v ) {
      for ( size_t u = 0; u < patch.getSizeU(); ++u ) {
        const size_t  p       = v * patch.getSizeU() + u;
        const int16_t d       = patch.getDepth( 0 )[p];
        const int16_t eomCode = patch.getDepthEnhancedDeltaD()[p];
        size_t        x;
        size_t        y;
        auto          indx = patch.patch2Canvas( u, v, width, height, x, y );
        assert( x < width && y < height );
        const size_t d0 = d;
        if ( params_.mapCountMinus1_ == 0 ) {  // one layer
          bool updateOccupancy = ( d < infiniteDepth ) && ( occupancyMap[indx] == 1 );
          if ( updateOccupancy ) {
            const size_t N      = params_.EOMFixBitCount_;
            int16_t      symbol = ( 1 << N ) - 1;
            symbol -= eomCode;
            // uint16_t nbBits = 0;
            //          for ( uint16_t i = 0; i < N; i++ )
            //            if ( eomCode & ( 1 << i ) ) numOfEOMpoints++;  //
            // nbBits++;
            if ( symbol < 0 ) { symbol = 0; }
            occupancyMap[indx] += symbol;
          }
        } else {
          const size_t d1      = patch.getDepth( 1 )[p];
          bool updateOccupancy = ( ( d < infiniteDepth ) && ( occupancyMap[indx] == 1 ) && ( ( d1 - d0 ) > 1 ) );
          if ( updateOccupancy ) {
            uint16_t bits       = d1 - d0 - 1;
            uint16_t eomExtract = eomCode & ( ~( ( ~0 ) << bits ) );
            uint16_t symbol     = ( ( ( 1 << bits ) - 1 ) - eomExtract );
            occupancyMap[indx] += symbol;
          }
        }
      }
    }  // u
  }    // v

  if ( !params_.absoluteD1_ || !params_.absoluteT1_ ) { fullOccupancyMap = occupancyMap; }
}

void PCCEncoder::adjustReferenceAtlasFrames( PCCContext& context, size_t tileIndex ) {
  for ( size_t frameIdx = 2; frameIdx < context.getFrames().size(); frameIdx++ ) {
    std::cout << ":::::---- adjusting reference frames for frame " << frameIdx << "\ttile " << tileIndex << std::endl;
    auto&                 tile         = context[frameIdx].getTile( tileIndex );
    double                dMinListDist = 0;
    std::vector<PCCPatch> bestPatchList;
    size_t                bestListIdx   = 0;
    double                dTempListDist = 0;
    std::vector<PCCPatch> tempPatchList;
    for ( size_t listIdx = 0; listIdx < context.getNumOfRefAtlasFrameList(); listIdx++ ) {
      dTempListDist = 0;
      tempPatchList.clear();
      double dTempListDist = adjustReferenceAtlasFrame( context, tile, tileIndex, listIdx, tempPatchList );
      if ( dTempListDist > dMinListDist ) {
        dMinListDist  = dTempListDist;
        bestListIdx   = listIdx;
        bestPatchList = tempPatchList;
      }
    }
    tile.setNumRefIdxActive( std::min( frameIdx, context.getSizeOfRefAtlasFrameList( bestListIdx ) ) );
    tile.setBestRefListIndexInAsps( bestListIdx );
    tile.setRefAfocList( context, bestListIdx );
    tile.getPatches() = bestPatchList;
  }  // frame
}

double PCCEncoder::adjustReferenceAtlasFrame( PCCContext&            context,
                                              PCCFrameContext&       tile,
                                              size_t                 tileIndex,
                                              size_t                 listIndex,
                                              std::vector<PCCPatch>& tempPatchList ) {
  tile.setRefAfocList( context, listIndex );
  PCCBitstream tempBitStream;
  auto         curPatches    = tile.getPatches();
  size_t       curPatchCount = curPatches.size();
  if ( curPatches.empty() ) { return -1; }
  vector<double> maxIOUList;
  maxIOUList.resize( curPatchCount, -1.0F );
  double sumMaxIOU = 0;
  // initialization
  size_t maxU0 = 0;
  size_t maxV0 = 0;
  size_t maxU1 = 0;
  size_t maxV1 = 0;
  size_t maxD1 = 0;
  size_t maxDD;
  for ( size_t patchIdx = 0; patchIdx < curPatchCount; patchIdx++ ) {
    maxU0 = ( std::max )( maxU0, curPatches[patchIdx].getU0() );
    maxV0 = ( std::max )( maxV0, curPatches[patchIdx].getV0() );
    maxU1 = ( std::max )( maxU0, curPatches[patchIdx].getU1() );
    maxV1 = ( std::max )( maxU0, curPatches[patchIdx].getV1() );
    maxD1 = ( std::max )( maxU0, curPatches[patchIdx].getD1() );
    maxDD = ( std::max )( maxU0, curPatches[patchIdx].getSizeD() );
  }
  auto bitMaxU0 = uint8_t( ceilLog2( uint32_t( maxU0 ) ) );
  auto bitMaxV0 = uint8_t( ceilLog2( uint32_t( maxV0 ) ) );
  auto bitMaxU1 = uint8_t( ceilLog2( uint32_t( maxU1 ) ) );
  auto bitMaxV1 = uint8_t( ceilLog2( uint32_t( maxV1 ) ) );
  auto bitMaxD1 = uint8_t( ceilLog2( uint32_t( maxD1 ) ) );
  auto bitMaxDD = uint8_t( ceilLog2( uint32_t( maxDD ) ) );

  const size_t max3DCoordinate =
      size_t( 1 ) << ( params_.geometry3dCoordinatesBitdepth_ + ( params_.additionalProjectionPlaneMode_ > 0 ) );
  for ( size_t curId = 0; curId < curPatchCount; curId++ ) {
    auto& curPatch = curPatches[curId];
    // intra
    float initSize = tempBitStream.size();
    tempBitStream.write( uint32_t( curPatch.getU0() ), bitMaxU0 );
    tempBitStream.write( uint32_t( curPatch.getV0() ), bitMaxV0 );
    tempBitStream.writeSvlc(
        int32_t( curId == 0 ? curPatch.getSizeU0() : curPatch.getSizeU0() - curPatches[curId - 1].getSizeU0() ) );
    tempBitStream.writeSvlc(
        int32_t( curId == 0 ? curPatch.getSizeV0() : curPatch.getSizeV0() - curPatches[curId - 1].getSizeV0() ) );
    tempBitStream.write( uint32_t( curPatch.getU1() ), bitMaxU1 );
    tempBitStream.write( uint32_t( curPatch.getV1() ), bitMaxV1 );
    tempBitStream.write( uint32_t( curPatch.getD1() ), bitMaxD1 );
    tempBitStream.write( uint32_t( curPatch.getSizeD() ), bitMaxDD );
    tempBitStream.write( uint32_t( curPatch.getViewId() ), 3 );
    if ( params_.useEightOrientations_ ) {
      tempBitStream.write( curPatch.getPatchOrientation(), 3 );
    } else {
      tempBitStream.write( curPatch.getPatchOrientation(), 1 );
    }
    if ( curPatch.getAxisOfAdditionalPlane() != 0u ) {
      tempBitStream.write( uint32_t( curPatch.getAxisOfAdditionalPlane() ), 1 );
    }
    float bitCostIntraA = tempBitStream.size();

    // inter
    if ( curPatch.getBestMatchIdx() != -1 ) {
      int32_t refPOC = tile.getRefAfoc( 0 );
      if ( refPOC < 0 ) break;
      auto& refPatch = context[refPOC].getTile( tileIndex ).getPatch( curPatch.getBestMatchIdx() );
      tempBitStream.writeSvlc( int32_t( static_cast<int64_t>( curPatch.getBestMatchIdx() ) - curId ) );  // approx
      tempBitStream.writeUvlc( int32_t( 0 ) );
      tempBitStream.writeSvlc( int32_t( curPatch.getU0() - refPatch.getU0() ) );
      tempBitStream.writeSvlc( int32_t( curPatch.getV0() - refPatch.getV0() ) );
      tempBitStream.writeSvlc( int32_t( curPatch.getSizeU0() - refPatch.getSizeU0() ) );
      tempBitStream.writeSvlc( int32_t( curPatch.getSizeV0() - refPatch.getSizeV0() ) );
      tempBitStream.writeSvlc( int32_t( curPatch.getU1() - refPatch.getU1() ) );
      tempBitStream.writeSvlc( int32_t( curPatch.getV1() - refPatch.getV1() ) );

      size_t        quantDD  = curPatch.getSizeD() == 0 ? 0 : ( ( curPatch.getSizeD() - 1 ) / params_.minLevel_ + 1 );
      size_t        prevQDD  = refPatch.getSizeD() == 0 ? 0 : ( ( refPatch.getSizeD() - 1 ) / params_.minLevel_ + 1 );
      const int64_t delta_dd = ( static_cast<int64_t>( quantDD ) ) - ( static_cast<int64_t>( prevQDD ) );
      tempBitStream.writeSvlc( int32_t( delta_dd ) );  // se(v)

      int32_t delta_d1 = 0;
      if ( curPatch.getProjectionMode() == 0 || !params_.absoluteD1_ ) {
        delta_d1 = ( ( curPatch.getD1() / params_.minLevel_ ) - ( refPatch.getD1() / params_.minLevel_ ) );
      } else {
        if ( curPatch.getAxisOfAdditionalPlane() == 0 ) {
          delta_d1 = ( max3DCoordinate - curPatch.getD1() ) / params_.minLevel_ -
                     ( max3DCoordinate - refPatch.getD1() ) / params_.minLevel_;
        } else {
          delta_d1 = ( ( max3DCoordinate << 1 ) - curPatch.getD1() ) / params_.minLevel_ -
                     ( ( max3DCoordinate << 1 ) - refPatch.getD1() ) / params_.minLevel_;
        }
      }
      tempBitStream.writeSvlc( delta_d1 );
    }
#ifdef CODEC_TRACE
    float bitCostInterA = ( curPatch.getBestMatchIdx() != -1 ) ? tempBitStream.size() : 0;
    float bitCostInter  = bitCostInterA - initSize;
#endif
    float bitCostIntra = bitCostIntraA - initSize;

    maxIOUList[curId] = 1 / bitCostIntra;
    curPatch.setBestMatchIdx( -1 );
  }

  // loop over refPicture in the list
  size_t sizeOfList = tile.getRefAfocListSize();
  for ( size_t refIdx = 0; refIdx < sizeOfList; refIdx++ ) {
    int32_t refPOC = tile.getRefAfoc( refIdx );
    if ( refPOC < 0 ) continue;
    auto& refPatches = context.getFrame( refPOC ).getTile( tileIndex ).getPatches();
    for ( size_t refPatchId = 0; refPatchId < refPatches.size(); refPatchId++ ) {
      // bestOrderPatches.clear();
      auto& refPatch   = refPatches[refPatchId];
      float maxIou     = 0.0F;
      int   bestCurIdx = -1;
      for ( size_t curId = 0; curId < curPatchCount; curId++ ) {
        auto& curPatch     = curPatches[curId];
        bool  bMatchingRef = refPatch.getViewId() == curPatch.getViewId() &&
                            refPatch.getPatchOrientation() == curPatch.getPatchOrientation();
        if ( bMatchingRef ) {
          float initSize = tempBitStream.size();
          tempBitStream.writeSvlc( int32_t( static_cast<int64_t>( refPatchId ) - curId ) );  // approx
          tempBitStream.writeUvlc( int32_t( refIdx ) );
          tempBitStream.writeSvlc( int32_t( curPatch.getU0() - refPatch.getU0() ) );
          tempBitStream.writeSvlc( int32_t( curPatch.getV0() - refPatch.getV0() ) );
          tempBitStream.writeSvlc( int32_t( curPatch.getSizeU0() - refPatch.getSizeU0() ) );
          tempBitStream.writeSvlc( int32_t( curPatch.getSizeV0() - refPatch.getSizeV0() ) );
          tempBitStream.writeSvlc( int32_t( curPatch.getU1() - refPatch.getU1() ) );
          tempBitStream.writeSvlc( int32_t( curPatch.getV1() - refPatch.getV1() ) );
          size_t quantDD = curPatch.getSizeD() == 0 ? 0 : ( ( curPatch.getSizeD() - 1 ) / params_.minLevel_ + 1 );
          size_t prevQDD = refPatch.getSizeD() == 0 ? 0 : ( ( refPatch.getSizeD() - 1 ) / params_.minLevel_ + 1 );
          const int64_t delta_dd = ( static_cast<int64_t>( quantDD ) ) - ( static_cast<int64_t>( prevQDD ) );
          tempBitStream.writeSvlc( int32_t( delta_dd ) );  // se(v)

          int32_t delta_d1 = 0;
          if ( curPatch.getProjectionMode() == 0 || !params_.absoluteD1_ ) {
            delta_d1 = ( ( curPatch.getD1() / params_.minLevel_ ) - ( refPatch.getD1() / params_.minLevel_ ) );
          } else {
            if ( curPatch.getAxisOfAdditionalPlane() == 0 ) {
              delta_d1 = ( max3DCoordinate - curPatch.getD1() ) / params_.minLevel_ -
                         ( max3DCoordinate - refPatch.getD1() ) / params_.minLevel_;
            } else {
              delta_d1 = ( ( max3DCoordinate << 1 ) - curPatch.getD1() ) / params_.minLevel_ -
                         ( ( max3DCoordinate << 1 ) - refPatch.getD1() ) / params_.minLevel_;
            }
          }
          tempBitStream.writeSvlc( delta_d1 );

          //      float bitCostInterA=tempBitStream.size();
          float bitCostInter = tempBitStream.size() - initSize;
          // float bitCostIntra=bitCostIntraA-bitCostInterA;
          // float areaOverlap=computeIOU( refRect, curRect );
          float iou = 1 / bitCostInter;

          if ( iou > maxIou ) {
            maxIou     = iou;
            bestCurIdx = curId;
          }

        }  // end of if (patch.viewId == cpatch.viewId).
      }
      if ( bestCurIdx >= 0 && maxIou > maxIOUList[bestCurIdx] ) {
        curPatches[bestCurIdx].setBestMatchIdx( refPatchId );    // the matched patch id in preivious frame.
        curPatches[bestCurIdx].setRefAtlasFrameIndex( refIdx );  // the matched patch id in preivious frame.
        curPatches[bestCurIdx].setPatchType( static_cast<uint8_t>( P_INTER ) );
        maxIOUList[bestCurIdx] = maxIou;
      }
    }  // refPatch

  }  // refIdx

  // no reordering!
  size_t numInterPredictedPatches = 0;
  for ( size_t patchIdx = 0; patchIdx < curPatchCount; patchIdx++ ) {
    if ( curPatches[patchIdx].getBestMatchIdx() != PCC_UNDEFINED_INDEX ) {
      curPatches[patchIdx].setPatchType( static_cast<uint8_t>( P_INTER ) );
      numInterPredictedPatches++;
      sumMaxIOU += maxIOUList[patchIdx];
    } else {
      curPatches[patchIdx].setPatchType( static_cast<uint8_t>( P_INTRA ) );
    }

    tempPatchList.push_back( curPatches[patchIdx] );
  }

  tile.setNumMatchedPatches( numInterPredictedPatches );
  return sumMaxIOU;
}

void PCCEncoder::spatialConsistencyPackFlexible( PCCFrameContext& tile,
                                                 PCCFrameContext& prevFrame,
                                                 int              packingStrategy,
                                                 size_t           presetWidth,
                                                 size_t           presetHeight,
                                                 int              safeguard,
                                                 bool             enablePointCloudPartitioning ) {
  auto& width   = tile.getWidth();
  auto& height  = tile.getHeight();
  auto& patches = tile.getPatches();

  auto& prevPatches = prevFrame.getPatches();
  if ( patches.empty() ) {
    if ( tile.getNumberOfRawPointsPatches() == 0 ) { return; }
    if ( tile.getUseRawPointsSeparateVideo() ) { return; }
    std::vector<bool> occupancyMap;
    size_t            occupancySizeU = presetWidth / params_.occupancyResolution_;
    size_t            occupancySizeV = presetHeight / params_.occupancyResolution_;
    if ( presetWidth == 0 || presetHeight == 0 ) {
      auto& rawPointsPatch       = tile.getRawPointsPatch( 0 );
      auto  rawPointsPatchBlocks = static_cast<size_t>(
          ceil( double( rawPointsPatch.sizeX() ) / ( params_.occupancyResolution_ * params_.occupancyResolution_ ) ) );
      if ( presetWidth == 0 ) occupancySizeU = params_.minimumImageWidth_ / params_.occupancyResolution_;
      if ( presetHeight == 0 )
        occupancySizeV = static_cast<size_t>( ceil( double( rawPointsPatchBlocks ) / occupancySizeU ) );
    }
    occupancyMap.resize( occupancySizeU * occupancySizeV );
    if ( tile.getNumberOfRawPointsPatches() > 0 && !tile.getUseRawPointsSeparateVideo() ) {
      packRawPointsPatch( tile, occupancyMap, width, height, occupancySizeU, occupancySizeV, 0 );
    } else {
      if ( printDetailedInfo ) { printMap( occupancyMap, occupancySizeU, occupancySizeV ); }
    }
    if ( params_.enhancedOccupancyMapCode_ && !tile.getUseRawPointsSeparateVideo() ) {
      packEOMTexturePointsPatch( tile, occupancyMap, width, height, occupancySizeU, occupancySizeV, 0 );
    }
    if ( printDetailedInfo ) { printMap( occupancyMap, occupancySizeU, occupancySizeV ); }
    bool emptyTile = false;
    if ( height == 0 ) {
      emptyTile = true;
      height    = 64;
    }
    std::cout << "frame " << tile.getFrameIndex() << " tile " << tile.getTileIndex()
              << " spatialConsistencyPackFlexible(patchEmpty): actualImageSize " << width << " x " << height;
    if ( emptyTile )
      std::cout << " height adjusted" << std::endl;
    else
      std::cout << std::endl;
    return;
  }
#if 1
  if ( packingStrategy == 0 )
    std::sort( patches.begin(), patches.end() );
  else
#endif
    std::sort( patches.begin(), patches.end(), []( PCCPatch& a, PCCPatch& b ) { return a.gt( b ); } );
  int              id             = 0;
  size_t           occupancySizeU = presetWidth / params_.occupancyResolution_;
  size_t           occupancySizeV = ( std::max )( patches[0].getSizeU0(), patches[0].getSizeV0() );
  vector<PCCPatch> matchedPatches;
  vector<PCCPatch> tmpPatches;
  matchedPatches.clear();
  float  thresholdIOU    = 0.2F;
  size_t bestRefFrameIdx = 0;
  // main loop.
  for ( auto& patch : prevPatches ) {
    id++;
    float maxIou  = 0.0F;
    int   bestIdx = -1;
    int   cId     = 0;
    for ( auto& cpatch : patches ) {
      if ( ( patch.getViewId() == cpatch.getViewId() ) && ( cpatch.getBestMatchIdx() == InvalidPatchIndex ) &&
           ( patch.getLodScaleX() == cpatch.getLodScaleX() && patch.getLodScaleY() == cpatch.getLodScaleY() ) ) {
        patch.setPatchType( static_cast<uint8_t>( P_INTRA ) );
        Rect  rect  = Rect( patch.getU1(), patch.getV1(), patch.getSizeU(), patch.getSizeV() );
        Rect  crect = Rect( cpatch.getU1(), cpatch.getV1(), cpatch.getSizeU(), cpatch.getSizeV() );
        float iou   = computeIOU( rect, crect );
        if ( iou > maxIou ) {
          maxIou  = iou;
          bestIdx = cId;
        }
      }  // end of if (patch.viewId == cpatch.viewId).
      cId++;
    }

    if ( maxIou > thresholdIOU ) {
      // store the best match index
      patches[bestIdx].setBestMatchIdx( id - 1 );  // the matched patch id in preivious frame.
      patches[bestIdx].setPatchType( static_cast<uint8_t>( P_INTER ) );
      patches[bestIdx].setRefAtlasFrameIndex( bestRefFrameIdx );
      matchedPatches.push_back( patches[bestIdx] );
    }
  }

  // generate new patch order.
  vector<PCCPatch> newOrderPatches = matchedPatches;

  for ( auto patch : patches ) {
    assert( patch.getSizeU0() <= occupancySizeU );
    assert( patch.getSizeV0() <= occupancySizeV );
    if ( patch.getBestMatchIdx() == InvalidPatchIndex ) {
      patch.setPatchType( static_cast<uint8_t>( P_INTRA ) );
      newOrderPatches.push_back( patch );
    }
  }
  tile.setNumMatchedPatches( matchedPatches.size() );

  // remove the below logs when useless.
  if ( printDetailedInfo ) {
    std::cout << "patches.size:" << patches.size() << ",reOrderedPatches.size:" << newOrderPatches.size()
              << ",matchedpatches.size:" << tile.getNumMatchedPatches() << std::endl;
  }
  patches = newOrderPatches;
  if ( printDetailedInfo ) {
    std::cout << "Patch order:" << std::endl;
    for ( auto& patch : patches ) {
      std::cout << "Patch[" << patch.getIndex() << "]=(" << patch.getSizeU0() << "," << patch.getSizeV0() << ")"
                << std::endl;
    }
  }

  for ( auto& patch : patches ) { occupancySizeU = ( std::max )( occupancySizeU, patch.getSizeU0() + 1 ); }

  int numTilesHor = params_.numTilesHor_;
  int tileWidth   = occupancySizeU / numTilesHor;
  int tileHeight  = int( tileWidth * params_.tileHeightToWidthRatio_ );
  if ( params_.enablePointCloudPartitioning_ ) {
    std::cout << "frame " << tile.getFrameIndex() << " tilesize: " << tileWidth << "x" << tileHeight << std::endl;
  }
  occupancySizeV = ( occupancySizeV >= tileHeight ) ? occupancySizeV : tileHeight;

  width  = occupancySizeU * params_.occupancyResolution_;
  height = occupancySizeV * params_.occupancyResolution_;
  size_t maxOccupancyRow{0};

  int               numOrientations = packingStrategy == 0 ? 1 : ( params_.useEightOrientations_ ? 8 : 2 );
  std::vector<bool> occupancyMap;
  occupancyMap.resize( occupancySizeU * occupancySizeV, false );
  for ( auto& patch : patches ) {
    assert( patch.getSizeU0() <= occupancySizeU );
    assert( patch.getSizeV0() <= occupancySizeV );
    bool  locationFound = false;
    auto& occupancy     = patch.getOccupancy();
    while ( !locationFound ) {
      if ( patch.getBestMatchIdx() != InvalidPatchIndex ) {
        patch.getPatchOrientation() = prevPatches[patch.getBestMatchIdx()].getPatchOrientation();
        // try to place on the same position as the matched patch
        patch.getU0() = prevPatches[patch.getBestMatchIdx()].getU0();
        patch.getV0() = prevPatches[patch.getBestMatchIdx()].getV0();
        if ( patch.checkFitPatchCanvas( occupancyMap, occupancySizeU, occupancySizeV, params_.lowDelayEncoding_ ) ) {
          locationFound = true;
          if ( printDetailedInfo ) {
            std::cout << "Maintained orientation " << patch.getPatchOrientation() << " for matched patch "
                      << patch.getIndex() << " in the same position (" << patch.getU0() << "," << patch.getV0() << ")"
                      << std::endl;
          }
        }
        // if the patch couldn't fit, try to fit the patch in the top left
        // position
        for ( int v = 0; v <= occupancySizeV && !locationFound; ++v ) {
          for ( int u = 0; u <= occupancySizeU && !locationFound; ++u ) {
            patch.getU0() = u;
            patch.getV0() = v;
            if ( patch.checkFitPatchCanvas( occupancyMap, occupancySizeU, occupancySizeV, params_.lowDelayEncoding_,
                                            safeguard ) ) {
              locationFound = true;
              if ( printDetailedInfo ) {
                std::cout << "Maintained orientation " << patch.getPatchOrientation() << " for matched patch "
                          << patch.getIndex() << " (" << u << "," << v << ")" << std::endl;
              }
            }
          }
        }
      } else {
        // best effort
        for ( size_t v = 0; v < occupancySizeV && !locationFound; ++v ) {
          for ( size_t u = 0; u < occupancySizeU && !locationFound; ++u ) {
            patch.getU0() = u;
            patch.getV0() = v;
            for ( size_t orientationIdx = 0; orientationIdx < numOrientations && !locationFound; orientationIdx++ ) {
              if ( packingStrategy == 0 )
                patch.getPatchOrientation() = PATCH_ORIENTATION_DEFAULT;
              else {
                if ( patch.getSizeU0() > patch.getSizeV0() ) {
                  patch.getPatchOrientation() = orientation_horizontal[orientationIdx];
                } else {
                  patch.getPatchOrientation() = orientation_vertical[orientationIdx];
                }
              }
              if ( patch.checkFitPatchCanvas( occupancyMap, occupancySizeU, occupancySizeV, params_.lowDelayEncoding_,
                                              safeguard ) ) {
                locationFound = true;
                if ( printDetailedInfo ) {
                  std::cout << "Orientation " << patch.getPatchOrientation() << " selected for unmatched patch "
                            << patch.getIndex() << " (" << u << "," << v << ")" << std::endl;
                }
              }
            }
          }
        }
      }
      if ( !locationFound ) {
        occupancySizeV *= 2;
        occupancyMap.resize( occupancySizeU * occupancySizeV );
      }
    }
    for ( size_t v0 = 0; v0 < patch.getSizeV0(); ++v0 ) {
      for ( size_t u0 = 0; u0 < patch.getSizeU0(); ++u0 ) {
        int coord = patch.patchBlock2CanvasBlock( u0, v0, occupancySizeU, occupancySizeV );
        if ( params_.lowDelayEncoding_ ) {
          occupancyMap[coord] = true;
        } else {
          occupancyMap[coord] = occupancyMap[coord] || occupancy[v0 * patch.getSizeU0() + u0];
        }
      }
    }
    if ( !( patch.isPatchDimensionSwitched() ) ) {
      height          = ( std::max )( height, ( patch.getV0() + patch.getSizeV0() ) * patch.getOccupancyResolution() );
      width           = ( std::max )( width, ( patch.getU0() + patch.getSizeU0() ) * patch.getOccupancyResolution() );
      maxOccupancyRow = ( std::max )( maxOccupancyRow, ( patch.getV0() + patch.getSizeV0() ) );
    } else {
      height          = ( std::max )( height, ( patch.getV0() + patch.getSizeU0() ) * patch.getOccupancyResolution() );
      width           = ( std::max )( width, ( patch.getU0() + patch.getSizeV0() ) * patch.getOccupancyResolution() );
      maxOccupancyRow = ( std::max )( maxOccupancyRow, ( patch.getV0() + patch.getSizeU0() ) );
    }
  }

  if ( tile.getNumberOfRawPointsPatches() > 0 && !tile.getUseRawPointsSeparateVideo() ) {
    packRawPointsPatch( tile, occupancyMap, width, height, occupancySizeU, occupancySizeV, maxOccupancyRow );
  } else {
    if ( printDetailedInfo ) { printMap( occupancyMap, occupancySizeU, occupancySizeV ); }
  }
  if ( params_.enhancedOccupancyMapCode_ && !tile.getUseRawPointsSeparateVideo() ) {
    packEOMTexturePointsPatch( tile, occupancyMap, width, height, occupancySizeU, occupancySizeV, maxOccupancyRow );
  }
  if ( printDetailedInfo ) { printMap( occupancyMap, occupancySizeU, occupancySizeV ); }
  std::cout << "frame " << tile.getFrameIndex() << " tile " << tile.getTileIndex()
            << " spatialConsistencyPackFlexible: actualImageSize " << width << " x " << height << std::endl;
}

void PCCEncoder::spatialConsistencyPackTetris( PCCFrameContext& frame,
                                               PCCFrameContext& prevFrame,
                                               size_t           presetWidth,
                                               size_t           presetHeight,
                                               int              safeguard ) {
  auto& width   = frame.getWidth();
  auto& height  = frame.getHeight();
  auto& patches = frame.getPatches();

  auto& prevPatches = prevFrame.getPatches();
  if ( patches.empty() ) { return; }
  std::sort( patches.begin(), patches.end(), []( PCCPatch& a, PCCPatch& b ) { return a.gt( b ); } );
  int              id             = 0;
  size_t           occupancySizeU = presetWidth / params_.occupancyResolution_;
  size_t           occupancySizeV = ( std::max )( patches[0].getSizeU0(), patches[0].getSizeV0() );
  vector<PCCPatch> matchedPatches;
  vector<PCCPatch> tmpPatches;
  matchedPatches.clear();
  float thresholdIOU = 0.2F;

  // main loop.
  for ( auto& patch : prevPatches ) {
    assert( patch.getSizeU0() <= occupancySizeU );
    assert( patch.getSizeV0() <= occupancySizeV );
    id++;
    float maxIou  = 0.0;
    int   bestIdx = -1;
    int   cId     = 0;
    for ( auto& cpatch : patches ) {
      if ( ( patch.getViewId() == cpatch.getViewId() ) && ( cpatch.getBestMatchIdx() == -1 ) &&
           ( patch.getLodScaleX() == cpatch.getLodScaleX() && patch.getLodScaleY() == cpatch.getLodScaleY() ) ) {
        Rect  rect  = Rect( patch.getU1(), patch.getV1(), patch.getSizeU(), patch.getSizeV() );
        Rect  crect = Rect( cpatch.getU1(), cpatch.getV1(), cpatch.getSizeU(), cpatch.getSizeV() );
        float iou   = computeIOU( rect, crect );
        if ( iou > maxIou ) {
          maxIou  = iou;
          bestIdx = cId;
        }
      }  // end of if (patch.viewId == cpatch.viewId).
      cId++;
    }

    if ( maxIou > thresholdIOU ) {
      // store the best match index
      patches[bestIdx].setBestMatchIdx( id - 1 );  // the matched patch id in preivious frame.
      patches[bestIdx].setPatchType( static_cast<uint8_t>( P_INTER ) );
      matchedPatches.push_back( patches[bestIdx] );
    }
  }

  // generate new patch order.
  vector<PCCPatch> newOrderPatches = matchedPatches;

  for ( auto patch : patches ) {
    assert( patch.getSizeU0() <= occupancySizeU );
    assert( patch.getSizeV0() <= occupancySizeV );
    if ( patch.getBestMatchIdx() == -1 ) { newOrderPatches.push_back( patch ); }
  }
  frame.setNumMatchedPatches( matchedPatches.size() );

  // remove the below logs when useless.
  if ( printDetailedInfo ) {
    std::cout << "patches.size:" << patches.size() << ",reOrderedPatches.size:" << newOrderPatches.size()
              << ",matchedpatches.size:" << frame.getNumMatchedPatches() << std::endl;
  }
  patches = newOrderPatches;
  if ( printDetailedInfo ) {
    std::cout << "Patch order:" << std::endl;
    for ( auto& patch : patches ) {
      std::cout << "Patch[" << patch.getIndex() << "]=(" << patch.getSizeU0() << "," << patch.getSizeV0() << ")"
                << std::endl;
    }
  }

  for ( auto& patch : patches ) { occupancySizeU = ( std::max )( occupancySizeU, patch.getSizeU0() + 1 ); }

  width  = occupancySizeU * params_.occupancyResolution_;
  height = occupancySizeV * params_.occupancyResolution_;
  size_t maxOccupancyRow{0};

  std::vector<bool> occupancyMap;
  occupancyMap.resize( occupancySizeU * occupancySizeV, false );
  std::vector<int> horizon;
  horizon.resize( occupancySizeU, 0 );

  for ( auto& patch : patches ) {
    assert( patch.getSizeU0() <= occupancySizeU );
    assert( patch.getSizeV0() <= occupancySizeV );
    auto& occupancy = patch.getOccupancy();
    // getting the horizons using the rotation 0 position
    std::vector<int> top_horizon;
    std::vector<int> bottom_horizon;
    std::vector<int> right_horizon;
    std::vector<int> left_horizon;
    patch.get_patch_horizons( top_horizon, bottom_horizon, right_horizon, left_horizon );

    bool locationFound = false;
    while ( !locationFound ) {
      int    best_wasted_space = ( std::numeric_limits<int>::max )();
      size_t best_u;
      size_t best_v;
      int    best_orientation;
      if ( patch.getBestMatchIdx() != -1 ) {
        patch.getPatchOrientation() = prevPatches[patch.getBestMatchIdx()].getPatchOrientation();
        best_orientation            = patch.getPatchOrientation();
        // spiral search to find the closest available position
        int x   = 0;
        int y   = 0;
        int end = ( std::max )( occupancySizeU, occupancySizeV ) * ( std::max )( occupancySizeU, occupancySizeV ) * 4;
        for ( int i = 0; i < end && !locationFound; ++i ) {
          // Translate coordinates and mask them out.
          int xp = x + prevPatches[patch.getBestMatchIdx()].getU0();
          int yp = y + prevPatches[patch.getBestMatchIdx()].getV0();
          if ( printDetailedInfo ) { std::cout << "Testing position (" << xp << ',' << yp << ')' << std::endl; }
          if ( xp >= 0 && xp < occupancySizeU && yp >= 0 && yp < occupancySizeV ) {
            patch.getU0() = xp;
            patch.getV0() = yp;
            if ( patch.checkFitPatchCanvas( occupancyMap, occupancySizeU, occupancySizeV, params_.lowDelayEncoding_,
                                            safeguard ) ) {
              locationFound = true;
              best_u        = xp;
              best_v        = yp;
              if ( printDetailedInfo ) {
                std::cout << "Maintained orientation " << patch.getPatchOrientation() << " for matched patch "
                          << patch.getIndex() << " in new position (" << xp << "," << yp << ")" << std::endl;
              }
            }
          }
          if ( abs( x ) <= abs( y ) && ( x != y || x >= 0 ) ) {
            x += ( ( y >= 0 ) ? 1 : -1 );
          } else {
            y += ( ( x >= 0 ) ? -1 : 1 );
          }
        }
      } else {
        vector<int> orientation_values = {
            PATCH_ORIENTATION_DEFAULT, PATCH_ORIENTATION_SWAP,    PATCH_ORIENTATION_ROT180,
            PATCH_ORIENTATION_MIRROR,  PATCH_ORIENTATION_MROT180, PATCH_ORIENTATION_ROT270,
            PATCH_ORIENTATION_MROT90,  PATCH_ORIENTATION_ROT90};  // favoring vertical orientation
        int numOrientations = params_.useEightOrientations_ ? 8 : 2;
        // tetris packing
        for ( size_t u = 0; u < occupancySizeU; ++u ) {
          for ( size_t v = 0; v < occupancySizeV; ++v ) {
            patch.getU0() = u;
            patch.getV0() = v;
            for ( size_t orientationIdx = 0; orientationIdx < numOrientations; orientationIdx++ ) {
              patch.getPatchOrientation() = orientation_values[orientationIdx];
              if ( !patch.isPatchLocationAboveHorizon( horizon, top_horizon, bottom_horizon, right_horizon,
                                                       left_horizon ) ) {
                if ( printDetailedInfo ) {
                  std::cout << "(" << u << "," << v << "|" << patch.getPatchOrientation() << ") above horizon"
                            << std::endl;
                }
                continue;
              }
              if ( patch.checkFitPatchCanvas( occupancyMap, occupancySizeU, occupancySizeV, params_.lowDelayEncoding_,
                                              safeguard ) ) {
                // now calculate the wasted space
                int wasted_space =
                    patch.calculate_wasted_space( horizon, top_horizon, bottom_horizon, right_horizon, left_horizon );
                if ( wasted_space < best_wasted_space ) {
                  best_wasted_space = wasted_space;
                  best_u            = u;
                  best_v            = v;
                  best_orientation  = patch.getPatchOrientation();
                  locationFound     = true;
                }
              }
            }
          }
        }
      }
      if ( !locationFound ) {
        occupancySizeV *= 2;
        occupancyMap.resize( occupancySizeU * occupancySizeV );
      } else {
        // select the best position and orientation
        patch.getU0()               = best_u;
        patch.getV0()               = best_v;
        patch.getPatchOrientation() = best_orientation;
        if ( printDetailedInfo ) {
          std::cout << "Selected position (" << best_u << "," << best_v << ") and orientation " << best_orientation
                    << std::endl;
        }
        // update the horizon
        patch.update_horizon( horizon, top_horizon, bottom_horizon, right_horizon, left_horizon );
        // debugging
        if ( printDetailedInfo ) {
          std::cout << "New Horizon :[";
          for ( int i = 0; i < occupancySizeU; i++ ) { std::cout << horizon[i] << ","; }
          std::cout << "]" << std::endl;
        }
      }
    }
    for ( size_t v0 = 0; v0 < patch.getSizeV0(); ++v0 ) {
      for ( size_t u0 = 0; u0 < patch.getSizeU0(); ++u0 ) {
        int coord = patch.patchBlock2CanvasBlock( u0, v0, occupancySizeU, occupancySizeV );
        if ( params_.lowDelayEncoding_ ) {
          occupancyMap[coord] = true;
        } else {
          occupancyMap[coord] = occupancyMap[coord] || occupancy[v0 * patch.getSizeU0() + u0];
        }
      }
    }
    if ( !( patch.isPatchDimensionSwitched() ) ) {
      height          = ( std::max )( height, ( patch.getV0() + patch.getSizeV0() ) * patch.getOccupancyResolution() );
      width           = ( std::max )( width, ( patch.getU0() + patch.getSizeU0() ) * patch.getOccupancyResolution() );
      maxOccupancyRow = ( std::max )( maxOccupancyRow, ( patch.getV0() + patch.getSizeV0() ) );
    } else {
      height          = ( std::max )( height, ( patch.getV0() + patch.getSizeU0() ) * patch.getOccupancyResolution() );
      width           = ( std::max )( width, ( patch.getU0() + patch.getSizeV0() ) * patch.getOccupancyResolution() );
      maxOccupancyRow = ( std::max )( maxOccupancyRow, ( patch.getV0() + patch.getSizeU0() ) );
    }
    if ( printDetailedInfo ) { printMapTetris( occupancyMap, occupancySizeU, occupancySizeV, horizon ); }
  }

  if ( frame.getNumberOfRawPointsPatches() > 0 && !frame.getUseRawPointsSeparateVideo() ) {
    packRawPointsPatch( frame, occupancyMap, width, height, occupancySizeU, occupancySizeV, maxOccupancyRow );
  } else {
    if ( printDetailedInfo ) { printMap( occupancyMap, occupancySizeU, occupancySizeV ); }
  }
  if ( params_.enhancedOccupancyMapCode_ && !frame.getUseRawPointsSeparateVideo() ) {
    packEOMTexturePointsPatch( frame, occupancyMap, width, height, occupancySizeU, occupancySizeV, maxOccupancyRow );
  }
  if ( printDetailedInfo ) { printMap( occupancyMap, occupancySizeU, occupancySizeV ); }
  std::cout << "actualImageSize (spatialConsistencyPackTetris) " << width << " x " << height << std::endl;
}

// GTP - GLOBAL PATCH PACKING
void PCCEncoder::findMatchesForGlobalTetrisPacking( PCCFrameContext& tile, PCCFrameContext& prevFrame ) {
  auto& patches     = tile.getPatches();
  auto& prevPatches = prevFrame.getPatches();
  if ( patches.empty() ) { return; }
  // sort the patches so that the first match is done with the largest patch
  // first.
  std::sort( patches.begin(), patches.end(), []( PCCPatch& a, PCCPatch& b ) { return a.gt( b ); } );

  if ( &tile == &prevFrame ) {
    if ( printDetailedInfo ) {
      for ( int patchIdx = 0; patchIdx < patches.size(); patchIdx++ ) {
        auto& patch = patches[patchIdx];
        std::cout << "Sorted Patch[" << patchIdx << "]->";
        patch.getU0() = 0;
        patch.getV0() = 0;
        patch.print();
      }
    }
    // no point in doing matching, this is the same frame
    return;
  }

  vector<PCCPatch> matchedPatches;
  int              id = 0;
  matchedPatches.clear();
  float thresholdIOU = 0.2F;
  // main loop.
  for ( auto& patch : prevPatches ) {
    id++;
    float maxIou  = 0.0F;
    int   bestIdx = -1;
    int   cId     = 0;
    for ( auto& cpatch : patches ) {
      if ( ( patch.getViewId() == cpatch.getViewId() ) && ( cpatch.getBestMatchIdx() == InvalidPatchIndex ) &&
           ( patch.getLodScaleX() == cpatch.getLodScaleX() && patch.getLodScaleY() == cpatch.getLodScaleY() ) ) {
        Rect  rect  = Rect( patch.getU1(), patch.getV1(), patch.getSizeU(), patch.getSizeV() );
        Rect  crect = Rect( cpatch.getU1(), cpatch.getV1(), cpatch.getSizeU(), cpatch.getSizeV() );
        float iou   = computeIOU( rect, crect );
        if ( iou > maxIou ) {
          maxIou  = iou;
          bestIdx = cId;
        }
      }  // end of if (patch.viewId == cpatch.viewId).
      cId++;
    }
    if ( maxIou > thresholdIOU ) {
      // checking the size of the matched patches
      auto&  curPatch = patches[bestIdx];
      double area1    = curPatch.getSizeU0() * curPatch.getSizeV0();
      double area2    = patch.getSizeU0() * patch.getSizeV0();
      if ( ( ( area1 / area2 ) < params_.globalPackingStrategyThreshold_ ) ||
           ( ( area2 / area1 ) < params_.globalPackingStrategyThreshold_ ) ) {
        // this seems like an unlike mismatch, will break the chain here
        if ( printDetailedInfo ) {
          std::cout << "Removing the match because areas are too different:" << std::endl;
          std::cout << "elem.ID =" << curPatch.getIndex() << std::endl;
          std::cout << "elem.sizeU0 =" << curPatch.getSizeU0() << std::endl;
          std::cout << "elem.sizeV0 =" << curPatch.getSizeV0() << std::endl;
          std::cout << "area =" << area1 << std::endl;
          std::cout << "previous_elem.ID =" << patch.getIndex() << std::endl;
          std::cout << "elem.sizeU0 =" << patch.getSizeU0() << std::endl;
          std::cout << "elem.sizeV0 =" << patch.getSizeV0() << std::endl;
          std::cout << "area =" << area2 << std::endl;
        }
      } else {
        // store the best match index
        patches[bestIdx].setBestMatchIdx( id - 1 );  // the matched patch id in previous frame.
        matchedPatches.push_back( patches[bestIdx] );
      }
    }
  }
  tile.setNumMatchedPatches( matchedPatches.size() );

  vector<PCCPatch> newOrderPatches = matchedPatches;
  for ( const auto& patch : patches ) {
    if ( patch.getBestMatchIdx() == InvalidPatchIndex ) { newOrderPatches.push_back( patch ); }
  }
  tile.setNumMatchedPatches( matchedPatches.size() );

  patches = newOrderPatches;
  if ( printDetailedInfo ) {
    for ( int patchIdx = 0; patchIdx < patches.size(); patchIdx++ ) {
      auto& patch = patches[patchIdx];
      if ( patchIdx < tile.getNumMatchedPatches() ) {
        std::cout << "Matched (refPatch[" << patches[patchIdx].getBestMatchIdx()
                  << "]=" << prevPatches[patches[patchIdx].getBestMatchIdx()].getIndex() << ") Patch[" << patchIdx
                  << "]->";
      } else {
        std::cout << "Unmatched Patch[" << patchIdx << "]->";
      }
      patch.getU0() = 0;
      patch.getV0() = 0;
      patch.print();
    }
  }
}

void PCCEncoder::doGlobalTetrisPacking( PCCContext& context,
                                        size_t      tileIndex,
                                        size_t      frameWidth,
                                        size_t      frameHeight,
                                        int         firstFrame,
                                        int         lastFramePlus1 ) {
  struct doubleLinkedPatchElement {
    pcc::PCCPatch*    elem{nullptr};
    int32_t           nextElemPos{-1};
    int32_t           prevElemPos{-1};
    int32_t           weight{0};
    pcc::PCCPatch     globalElem;
    std::vector<bool> globalOccupancyMap;

    doubleLinkedPatchElement() = default;
    doubleLinkedPatchElement( pcc::PCCPatch* patch ) :
        elem( patch ),
        prevElemPos( patch->getBestMatchIdx() ),
        weight( patch->getBestMatchIdx() >= 0 ? 1 : 0 ) {}
    bool gt( const doubleLinkedPatchElement& rhs ) {
      // setting the largest dimension
      if ( weight > rhs.weight ) { return true; }
      if ( weight < rhs.weight ) { return false; }
      { return elem->gt( rhs.elem[0] ); }
    }
  };

  std::vector<std::vector<doubleLinkedPatchElement>> patchMatrix;
  std::vector<std::vector<int32_t>>                  patchMatrixSortedIndexes;
  // creating the doubled linked list
  patchMatrix.resize( context.size() );
  patchMatrixSortedIndexes.resize( context.size() );
  auto& frames  = context.getFrames();
  int   gofSize = ( params_.globalPackingStrategyGOF_ == 0 ) ? frames.size() : params_.globalPackingStrategyGOF_;
  int   numGof  = ( frames.size() + gofSize / 2 ) / gofSize;
  for ( int gofIdx = 0; gofIdx < numGof; gofIdx++ ) {
    int frStart = ( gofIdx ) * ( gofSize );
    int frEnd   = ( gofIdx + 1 ) * ( gofSize );
    if ( gofIdx + 1 == numGof ) { frEnd = frames.size(); }
    for ( size_t frameIdx = frStart; frameIdx < frEnd; frameIdx++ ) {
      auto& patches = frames[frameIdx].getTile( tileIndex ).getPatches();
      patchMatrix[frameIdx].resize( patches.size() );
      patchMatrixSortedIndexes[frameIdx].resize( patches.size() );
      for ( size_t patchIdx = 0; patchIdx < patches.size(); patchIdx++ ) {
        doubleLinkedPatchElement elem( &patches[patchIdx] );
        if ( frameIdx == frStart ) {
          elem.prevElemPos = -1;  // break the link between GOFs
          if ( params_.globalPackingStrategyReset_ ) { elem.elem->setBestMatchIdx( InvalidPatchIndex ); }
        }
        patchMatrix[frameIdx][patchIdx]              = elem;
        patchMatrixSortedIndexes[frameIdx][patchIdx] = patchIdx;
      }
    }
    // now go from back to front and update the nextElemPos and the weight
    for ( int frameIdx = frEnd - 1; frameIdx >= frStart; frameIdx-- ) {
      for ( size_t patchIdx = 0; patchIdx < patchMatrix[frameIdx].size(); patchIdx++ ) {
        // update the list
        if ( patchMatrix[frameIdx][patchIdx].prevElemPos >= 0 ) {
          patchMatrix[frameIdx - 1][patchMatrix[frameIdx][patchIdx].prevElemPos].nextElemPos = patchIdx;
        }
        // update the weight, the global occupancy map, and the patch size
        if ( patchMatrix[frameIdx][patchIdx].nextElemPos >= 0 ) {
          // new weight
          patchMatrix[frameIdx][patchIdx].weight +=
              patchMatrix[frameIdx + 1][patchMatrix[frameIdx][patchIdx].nextElemPos].weight + 1;
          // new occupancy map
          size_t curU1                  = patchMatrix[frameIdx][patchIdx].elem->getU1();
          size_t curV1                  = patchMatrix[frameIdx][patchIdx].elem->getV1();
          size_t curSizeU0              = patchMatrix[frameIdx][patchIdx].elem->getSizeU0();
          size_t curSizeV0              = patchMatrix[frameIdx][patchIdx].elem->getSizeV0();
          size_t curOccupancyResolution = patchMatrix[frameIdx][patchIdx].elem->getOccupancyResolution();

          auto&  nextGlobalElem = patchMatrix[frameIdx + 1][patchMatrix[frameIdx][patchIdx].nextElemPos].globalElem;
          size_t nextU1         = nextGlobalElem.getU1();
          size_t nextV1         = nextGlobalElem.getV1();
          size_t nextSizeU0     = nextGlobalElem.getSizeU0();
          size_t nextSizeV0     = nextGlobalElem.getSizeV0();
          size_t nextOccupancyResolution = curOccupancyResolution;

          auto& curGlobalElem   = patchMatrix[frameIdx][patchIdx].globalElem;
          curGlobalElem.getU1() = min( curU1, nextU1 );
          curGlobalElem.getV1() = min( curV1, nextV1 );

          curGlobalElem.getSizeU0() = ( max( ( curSizeU0 - 1 ) * curOccupancyResolution + curU1,
                                             ( nextSizeU0 - 1 ) * nextOccupancyResolution + nextU1 ) -
                                        curGlobalElem.getU1() ) /
                                          curOccupancyResolution +
                                      1;
          curGlobalElem.getSizeV0() = ( max( ( curSizeV0 - 1 ) * curOccupancyResolution + curV1,
                                             ( nextSizeV0 - 1 ) * nextOccupancyResolution + nextV1 ) -
                                        curGlobalElem.getV1() ) /
                                          curOccupancyResolution +
                                      1;

          curGlobalElem.getOccupancy().resize( curGlobalElem.getSizeU0() * curGlobalElem.getSizeV0(), false );
          // copy the global occupancy map from next patch
          for ( size_t v = 0; v < nextSizeV0; v++ ) {
            for ( size_t u = 0; u < nextSizeU0; u++ ) {
              size_t posGlobal =
                  ( nextU1 - curGlobalElem.getU1() ) / curOccupancyResolution + u +
                  curGlobalElem.getSizeU0() * ( v + ( nextV1 - curGlobalElem.getV1() ) / curOccupancyResolution );
              size_t posNext                          = u + nextSizeU0 * ( v );
              curGlobalElem.getOccupancy()[posGlobal] = nextGlobalElem.getOccupancy()[posNext];
            }
          }
          // copy the global occupancy map from current patch
          for ( size_t v = 0; v < curSizeV0; v++ ) {
            for ( size_t u = 0; u < curSizeU0; u++ ) {
              size_t posGlobal =
                  ( curU1 - curGlobalElem.getU1() ) / curOccupancyResolution + u +
                  curGlobalElem.getSizeU0() * ( v + ( curV1 - curGlobalElem.getV1() ) / curOccupancyResolution );
              size_t posCur                           = u + curSizeU0 * ( v );
              curGlobalElem.getOccupancy()[posGlobal] = curGlobalElem.getOccupancy()[posGlobal] ||
                                                        patchMatrix[frameIdx][patchIdx].elem->getOccupancy()[posCur];
            }
          }
          curGlobalElem.getOccupancyResolution() = curOccupancyResolution;

          if ( printDetailedInfo ) {
            std::cout << "Next Global Element" << std::endl;
            std::cout << "patchMatrix[" << frameIdx + 1 << "][" << patchMatrix[frameIdx][patchIdx].nextElemPos
                      << "].globalElem.U1 =" << nextU1 << std::endl;
            std::cout << "patchMatrix[" << frameIdx + 1 << "][" << patchMatrix[frameIdx][patchIdx].nextElemPos
                      << "].globalElem.V1 =" << nextV1 << std::endl;
            printMap( nextGlobalElem.getOccupancy(), nextGlobalElem.getSizeU0(), nextGlobalElem.getSizeV0() );

            std::cout << "Current Element" << std::endl;
            std::cout << "patchMatrix[" << frameIdx << "][" << patchIdx
                      << "].elem.ID =" << patchMatrix[frameIdx][patchIdx].elem->getIndex() << std::endl;
            std::cout << "patchMatrix[" << frameIdx << "][" << patchIdx
                      << "].elem.U1 =" << patchMatrix[frameIdx][patchIdx].elem->getU1() << std::endl;
            std::cout << "patchMatrix[" << frameIdx << "][" << patchIdx
                      << "].elem.V1 =" << patchMatrix[frameIdx][patchIdx].elem->getV1() << std::endl;
            printMap( patchMatrix[frameIdx][patchIdx].elem->getOccupancy(),
                      patchMatrix[frameIdx][patchIdx].elem->getSizeU0(),
                      patchMatrix[frameIdx][patchIdx].elem->getSizeV0() );

            std::cout << "Current Global Element" << std::endl;
            std::cout << "patchMatrix[" << frameIdx << "][" << patchIdx
                      << "].weight =" << patchMatrix[frameIdx][patchIdx].weight << std::endl;
            std::cout << "patchMatrix[" << frameIdx << "][" << patchIdx
                      << "].nextElemPos =" << patchMatrix[frameIdx][patchIdx].nextElemPos << std::endl;
            std::cout << "patchMatrix[" << frameIdx << "][" << patchIdx
                      << "].prevElemPos =" << patchMatrix[frameIdx][patchIdx].prevElemPos << std::endl;
            std::cout << "patchMatrix[" << frameIdx << "][" << patchIdx
                      << "].globalElem.getU1 =" << patchMatrix[frameIdx][patchIdx].globalElem.getU1() << std::endl;
            std::cout << "patchMatrix[" << frameIdx << "][" << patchIdx
                      << "].globalElem.getV1 =" << patchMatrix[frameIdx][patchIdx].globalElem.getV1() << std::endl;
            printMap( patchMatrix[frameIdx][patchIdx].globalElem.getOccupancy(),
                      patchMatrix[frameIdx][patchIdx].globalElem.getSizeU0(),
                      patchMatrix[frameIdx][patchIdx].globalElem.getSizeV0() );
          }

        } else {
          // first element in the chain, global parameters are equal to current
          // element
          auto& curGlobalElem   = patchMatrix[frameIdx][patchIdx].globalElem;
          curGlobalElem.getU1() = patchMatrix[frameIdx][patchIdx].elem->getU1();
          curGlobalElem.getV1() = patchMatrix[frameIdx][patchIdx].elem->getV1();

          curGlobalElem.getSizeU0() = patchMatrix[frameIdx][patchIdx].elem->getSizeU0();
          curGlobalElem.getSizeV0() = patchMatrix[frameIdx][patchIdx].elem->getSizeV0();

          curGlobalElem.getOccupancy().resize( curGlobalElem.getSizeU0() * curGlobalElem.getSizeV0() );
          // copy the global occupancy map from current patch
          size_t curSizeU0 = patchMatrix[frameIdx][patchIdx].elem->getSizeU0();
          size_t curSizeV0 = patchMatrix[frameIdx][patchIdx].elem->getSizeV0();
          for ( size_t v = 0; v < curSizeV0; v++ ) {
            for ( size_t u = 0; u < curSizeU0; u++ ) {
              size_t posCur = u + curSizeU0 * ( v );
              patchMatrix[frameIdx][patchIdx].globalElem.getOccupancy()[posCur] =
                  patchMatrix[frameIdx][patchIdx].elem->getOccupancy()[posCur];
            }
          }
          curGlobalElem.getOccupancyResolution() = patchMatrix[frameIdx][patchIdx].elem->getOccupancyResolution();
          if ( printDetailedInfo ) {
            std::cout << "patchMatrix[" << frameIdx << "][" << patchIdx
                      << "].elem.ID =" << patchMatrix[frameIdx][patchIdx].elem->getIndex() << std::endl;
            std::cout << "patchMatrix[" << frameIdx << "][" << patchIdx
                      << "].elem.U1 =" << patchMatrix[frameIdx][patchIdx].elem->getU1() << std::endl;
            std::cout << "patchMatrix[" << frameIdx << "][" << patchIdx
                      << "].elem.V1 =" << patchMatrix[frameIdx][patchIdx].elem->getV1() << std::endl;
            std::cout << "patchMatrix[" << frameIdx << "][" << patchIdx
                      << "].elem.sizeU0 =" << patchMatrix[frameIdx][patchIdx].elem->getSizeU0() << std::endl;
            std::cout << "patchMatrix[" << frameIdx << "][" << patchIdx
                      << "].elem.sizeV0 =" << patchMatrix[frameIdx][patchIdx].elem->getSizeV0() << std::endl;
            std::cout << "patchMatrix[" << frameIdx << "][" << patchIdx
                      << "].weight =" << patchMatrix[frameIdx][patchIdx].weight << std::endl;
            std::cout << "patchMatrix[" << frameIdx << "][" << patchIdx
                      << "].nextElemPos =" << patchMatrix[frameIdx][patchIdx].nextElemPos << std::endl;
            std::cout << "patchMatrix[" << frameIdx << "][" << patchIdx
                      << "].prevElemPos =" << patchMatrix[frameIdx][patchIdx].prevElemPos << std::endl;
            std::cout << "patchMatrix[" << frameIdx << "][" << patchIdx
                      << "].globalElem.getU1 =" << patchMatrix[frameIdx][patchIdx].globalElem.getU1() << std::endl;
            std::cout << "patchMatrix[" << frameIdx << "][" << patchIdx
                      << "].globalElem.getV1 =" << patchMatrix[frameIdx][patchIdx].globalElem.getV1() << std::endl;
            printMap( patchMatrix[frameIdx][patchIdx].globalElem.getOccupancy(),
                      patchMatrix[frameIdx][patchIdx].globalElem.getSizeU0(),
                      patchMatrix[frameIdx][patchIdx].globalElem.getSizeV0() );
          }
        }
      }
    }
    // packing for each frame
    size_t occupancySizeU = frameWidth / params_.occupancyResolution_;
    size_t occupancySizeV = ( std::max )( patchMatrix[0][0].elem->getSizeU0(), patchMatrix[0][0].elem->getSizeV0() );
    for ( size_t frameIdx = frStart; frameIdx < frEnd; frameIdx++ ) {
      auto& width  = frames[frameIdx].getTile( tileIndex ).getWidth();
      auto& height = frames[frameIdx].getTile( tileIndex ).getHeight();
      // sorting the list
      vector<PCCPatch> sortedPatches;
      auto&            patchUnsorted = patchMatrix[frameIdx];
      if ( frameIdx == frStart ) {
        sort( patchMatrixSortedIndexes[frameIdx].begin(), patchMatrixSortedIndexes[frameIdx].end(),
              [&patchUnsorted]( size_t i1, size_t i2 ) { return patchUnsorted[i1].gt( patchUnsorted[i2] ); } );
        if ( printDetailedInfo ) {
          for ( int patchIdx = 0; patchIdx < patchMatrixSortedIndexes[frameIdx].size(); patchIdx++ ) {
            auto& patch = patchMatrix[frameIdx][patchMatrixSortedIndexes[frameIdx][patchIdx]];
            std::cout << " New order Patch[" << patchIdx << "]->[" << patchMatrixSortedIndexes[frameIdx][patchIdx]
                      << "] (" << patchMatrix[frameIdx][patchMatrixSortedIndexes[frameIdx][patchIdx]].weight << "):";
            patch.elem->print();
          }
        }
      } else {
        // have to maintain the same order of the matched patches from previous
        // frame
        // now sort the rest of the list according to the sort function of the
        // class
        if ( frames[frameIdx].getTile( tileIndex ).getNumMatchedPatches() !=
             patchMatrixSortedIndexes[frameIdx].size() ) {
          sort(
              patchMatrixSortedIndexes[frameIdx].begin() + frames[frameIdx].getTile( tileIndex ).getNumMatchedPatches(),
              patchMatrixSortedIndexes[frameIdx].end(),
              [&patchUnsorted]( size_t i1, size_t i2 ) { return patchUnsorted[i1].gt( patchUnsorted[i2] ); } );
        }
        if ( printDetailedInfo ) {
          for ( int patchIdx = 0; patchIdx < frames[frameIdx].getTile( tileIndex ).getNumMatchedPatches();
                patchIdx++ ) {
            auto& patch = patchMatrix[frameIdx][patchMatrixSortedIndexes[frameIdx][patchIdx]];
            std::cout << " New order Matched (" << patch.prevElemPos << ") Patch[" << patchIdx << "]->["
                      << patchMatrixSortedIndexes[frameIdx][patchIdx] << "] (" << patch.weight << "):";
            patch.elem->print();
          }
          for ( int patchIdx = frames[frameIdx].getTile( tileIndex ).getNumMatchedPatches();
                patchIdx < patchMatrixSortedIndexes[frameIdx].size(); patchIdx++ ) {
            auto& patch = patchMatrix[frameIdx][patchMatrixSortedIndexes[frameIdx][patchIdx]];
            std::cout << " New order Unmatched Patch[" << patchIdx << "]->["
                      << patchMatrixSortedIndexes[frameIdx][patchIdx] << "] (" << patch.weight << "):";
            patch.elem->print();
          }
        }
      }
      size_t maxOccupancyRow{0};

      vector<int>      orientation_horizontal;
      int              numOrientations;
      std::vector<int> horizon;

      orientation_horizontal.resize( 8 );
      if ( params_.packingStrategy_ == 2 ) {
        orientation_horizontal = {PATCH_ORIENTATION_DEFAULT, PATCH_ORIENTATION_SWAP,    PATCH_ORIENTATION_ROT180,
                                  PATCH_ORIENTATION_MIRROR,  PATCH_ORIENTATION_MROT180, PATCH_ORIENTATION_ROT270,
                                  PATCH_ORIENTATION_MROT90,  PATCH_ORIENTATION_ROT90};
      } else {
        // favoring horizontal orientations (that should be rotated)
        orientation_horizontal = {PATCH_ORIENTATION_SWAP,   PATCH_ORIENTATION_DEFAULT, PATCH_ORIENTATION_ROT270,
                                  PATCH_ORIENTATION_MROT90, PATCH_ORIENTATION_ROT90,   PATCH_ORIENTATION_ROT180,
                                  PATCH_ORIENTATION_MIRROR, PATCH_ORIENTATION_MROT180};
      }
      numOrientations = params_.packingStrategy_ == 0 ? 1 : ( params_.useEightOrientations_ ? 8 : 2 );

      std::vector<bool> occupancyMap;
      occupancyMap.resize( occupancySizeU * occupancySizeV, false );
      int indNextMatchedPatch = 0;
      // patch loop
      for ( int patchIdx = 0; patchIdx < patchMatrixSortedIndexes[frameIdx].size(); patchIdx++ ) {
        if ( printDetailedInfo ) {
          std::cout << "Processing patchMatrix[" << frameIdx << "][" << patchMatrixSortedIndexes[frameIdx][patchIdx]
                    << "]" << std::endl;
        }
        auto& curPatchElem  = patchMatrix[frameIdx][patchMatrixSortedIndexes[frameIdx][patchIdx]];
        auto& curGlobalElem = curPatchElem.globalElem;
        assert( curPatchElem.elem->getSizeU0() <= occupancySizeU );
        assert( curPatchElem.elem->getSizeV0() <= occupancySizeV );
        bool  locationFound = false;
        auto& occupancy     = curGlobalElem.getOccupancy();

        std::vector<int> top_horizon;
        std::vector<int> bottom_horizon;
        std::vector<int> right_horizon;
        std::vector<int> left_horizon;
        if ( params_.packingStrategy_ == 2 ) {
          // getting the horizons using the rotation 0 position
          curGlobalElem.get_patch_horizons( top_horizon, bottom_horizon, right_horizon, left_horizon );
        }

        while ( !locationFound ) {
          int    best_wasted_space = ( std::numeric_limits<int>::max )();
          size_t best_u;
          size_t best_v;
          int    best_orientation;
          if ( curPatchElem.prevElemPos != InvalidPatchIndex ) {
            // try to place on the same position as the matched global patch,
            // with the same orientation
            auto& previousGlobalElem            = patchMatrix[frameIdx - 1][curPatchElem.prevElemPos].globalElem;
            curGlobalElem.getPatchOrientation() = previousGlobalElem.getPatchOrientation();
            if ( !( curGlobalElem.isPatchDimensionSwitched() ) ) {
              curGlobalElem.getU0() =
                  previousGlobalElem.getU0() +
                  ( curGlobalElem.getU1() - previousGlobalElem.getU1() ) / curPatchElem.elem->getOccupancyResolution();
              curGlobalElem.getV0() =
                  previousGlobalElem.getV0() +
                  ( curGlobalElem.getV1() - previousGlobalElem.getV1() ) / curPatchElem.elem->getOccupancyResolution();
            } else {
              curGlobalElem.getU0() =
                  previousGlobalElem.getU0() +
                  ( curGlobalElem.getV1() - previousGlobalElem.getV1() ) / curPatchElem.elem->getOccupancyResolution();
              curGlobalElem.getV0() =
                  previousGlobalElem.getV0() +
                  ( curGlobalElem.getU1() - previousGlobalElem.getU1() ) / curPatchElem.elem->getOccupancyResolution();
            }
            if ( curGlobalElem.checkFitPatchCanvas( occupancyMap, occupancySizeU, occupancySizeV,
                                                    params_.lowDelayEncoding_ ) ) {
              locationFound = true;
              if ( params_.packingStrategy_ == 2 ) {
                // saving the best position for tetris packing
                best_u           = curGlobalElem.getU0();
                best_v           = curGlobalElem.getV0();
                best_orientation = curGlobalElem.getPatchOrientation();
              }
              // now put the local patch in the relative position
              if ( !( curGlobalElem.isPatchDimensionSwitched() ) ) {
                curPatchElem.elem->getU0() =
                    curGlobalElem.getU0() + ( curPatchElem.elem->getU1() - curGlobalElem.getU1() ) /
                                                curPatchElem.elem->getOccupancyResolution();
                curPatchElem.elem->getV0() =
                    curGlobalElem.getV0() + ( curPatchElem.elem->getV1() - curGlobalElem.getV1() ) /
                                                curPatchElem.elem->getOccupancyResolution();
              } else {
                curPatchElem.elem->getU0() =
                    curGlobalElem.getU0() + ( curPatchElem.elem->getV1() - curGlobalElem.getV1() ) /
                                                curPatchElem.elem->getOccupancyResolution();
                curPatchElem.elem->getV0() =
                    curGlobalElem.getV0() + ( curPatchElem.elem->getU1() - curGlobalElem.getU1() ) /
                                                curPatchElem.elem->getOccupancyResolution();
              }
              curPatchElem.elem->getPatchOrientation() = curGlobalElem.getPatchOrientation();
              curPatchElem.elem->setBestMatchIdx( std::distance(
                  patchMatrixSortedIndexes[frameIdx - 1].begin(),
                  std::find( patchMatrixSortedIndexes[frameIdx - 1].begin(),
                             patchMatrixSortedIndexes[frameIdx - 1].end(), curPatchElem.prevElemPos ) ) );
              if ( params_.packingStrategy_ != 2 ) {
                // if it is not tetris packing, we can save the position in the
                // list
                sortedPatches.push_back( *curPatchElem.elem );
                if ( printDetailedInfo ) {
                  std::cout << "Patch[" << curPatchElem.elem->getIndex() << "] maintained orientation "
                            << curPatchElem.elem->getPatchOrientation() << " for matched patch["
                            << curPatchElem.elem->getBestMatchIdx() << "] in the position ("
                            << curPatchElem.elem->getU0() << "," << curPatchElem.elem->getV0() << ")" << std::endl;
                }
                // if the element is the prediction of a patch in the next
                // frame, store the current order for the next
                // frame, to maintain the sequence
                if ( curPatchElem.nextElemPos >= 0 ) {
                  patchMatrixSortedIndexes[frameIdx + 1][indNextMatchedPatch++] = curPatchElem.nextElemPos;
                }
              }
            } else {
              std::cout << "Could not fit the global patch in the canvas, at "
                           "position ("
                        << curGlobalElem.getU0() << "," << curGlobalElem.getV0() << ") and orientation "
                        << curGlobalElem.getPatchOrientation() << " something went wrong " << std::endl;
              if ( printDetailedInfo ) {
                printMap( occupancyMap, occupancySizeU, occupancySizeV );
                printMap( occupancy, curGlobalElem.getSizeU0(), curGlobalElem.getSizeV0() );
              }
            }
          } else {
            if ( curPatchElem.elem->getBestMatchIdx() != InvalidPatchIndex ) {
              // this is a matched patch, but the first element of the list,
              if ( printDetailedInfo ) {
                std::cout << "Reference moved from position prevList[" << curPatchElem.elem->getBestMatchIdx() << "] ";
              }
              int32_t matchedIdx = std::distance(
                  patchMatrixSortedIndexes[frameIdx - 1].begin(),
                  std::find( patchMatrixSortedIndexes[frameIdx - 1].begin(),
                             patchMatrixSortedIndexes[frameIdx - 1].end(), curPatchElem.elem->getBestMatchIdx() ) );
              if ( printDetailedInfo ) {
                std::cout << "to position sortedPrevList[" << matchedIdx
                          << "]=" << patchMatrix[frameIdx - 1][matchedIdx].elem->getIndex() << std::endl;
              }
              // the orientation MUST be maintained,  but it can be placed
              // anywhere
              curGlobalElem.getPatchOrientation() = patchMatrix[frameIdx - 1][matchedIdx].elem->getPatchOrientation();
            }
            // best effort
            for ( size_t v = 0; v < occupancySizeV && ( ( params_.packingStrategy_ == 2 ) || !locationFound ); ++v ) {
              for ( size_t u = 0; u < occupancySizeU && ( ( params_.packingStrategy_ == 2 ) || !locationFound ); ++u ) {
                curGlobalElem.getU0() = u;
                curGlobalElem.getV0() = v;
                for ( size_t orientationIdx = 0;
                      orientationIdx < numOrientations && ( ( params_.packingStrategy_ == 2 ) || !locationFound );
                      orientationIdx++ ) {
                  if ( curPatchElem.elem->getBestMatchIdx() == InvalidPatchIndex ) {
                    if ( curGlobalElem.getSizeU0() > curGlobalElem.getSizeV0() ) {
                      curGlobalElem.getPatchOrientation() = orientation_horizontal[orientationIdx];
                    } else {
                      curGlobalElem.getPatchOrientation() = orientation_vertical[orientationIdx];
                    }
                  }
                  if ( params_.packingStrategy_ == 2 ) {
                    if ( !curGlobalElem.isPatchLocationAboveHorizon( horizon, top_horizon, bottom_horizon,
                                                                     right_horizon, left_horizon ) ) {
                      continue;
                    }
                  }
                  if ( curGlobalElem.checkFitPatchCanvas( occupancyMap, occupancySizeU, occupancySizeV,
                                                          params_.lowDelayEncoding_ ) ) {
                    if ( params_.packingStrategy_ == 2 ) {
                      // now calculate the wasted space
                      int wasted_space = curGlobalElem.calculate_wasted_space( horizon, top_horizon, bottom_horizon,
                                                                               right_horizon, left_horizon );
                      if ( wasted_space < best_wasted_space ) {
                        best_wasted_space = wasted_space;
                        best_u            = u;
                        best_v            = v;
                        best_orientation  = curGlobalElem.getPatchOrientation();
                        locationFound     = true;
                      }
                    } else {
                      locationFound = true;
                      if ( !( curGlobalElem.isPatchDimensionSwitched() ) ) {
                        curPatchElem.elem->getU0() =
                            curGlobalElem.getU0() + ( curPatchElem.elem->getU1() - curGlobalElem.getU1() ) /
                                                        curPatchElem.elem->getOccupancyResolution();
                        curPatchElem.elem->getV0() =
                            curGlobalElem.getV0() + ( curPatchElem.elem->getV1() - curGlobalElem.getV1() ) /
                                                        curPatchElem.elem->getOccupancyResolution();
                      } else {
                        curPatchElem.elem->getU0() =
                            curGlobalElem.getU0() + ( curPatchElem.elem->getV1() - curGlobalElem.getV1() ) /
                                                        curPatchElem.elem->getOccupancyResolution();
                        curPatchElem.elem->getV0() =
                            curGlobalElem.getV0() + ( curPatchElem.elem->getU1() - curGlobalElem.getU1() ) /
                                                        curPatchElem.elem->getOccupancyResolution();
                      }
                      curPatchElem.elem->getPatchOrientation() = curGlobalElem.getPatchOrientation();
                      if ( curPatchElem.elem->getBestMatchIdx() >= 0 ) {
                        curPatchElem.elem->setBestMatchIdx(
                            std::distance( patchMatrixSortedIndexes[frameIdx - 1].begin(),
                                           std::find( patchMatrixSortedIndexes[frameIdx - 1].begin(),
                                                      patchMatrixSortedIndexes[frameIdx - 1].end(),
                                                      curPatchElem.elem->getBestMatchIdx() ) ) );
                      }
                      sortedPatches.push_back( *curPatchElem.elem );
                      if ( printDetailedInfo ) {
                        if ( curPatchElem.elem->getBestMatchIdx() >= 0 ) {
                          std::cout << "Orientation " << curPatchElem.elem->getPatchOrientation()
                                    << " maintained for matched patch " << curPatchElem.elem->getIndex() << " ("
                                    << curPatchElem.elem->getU0() << "," << curPatchElem.elem->getV0()
                                    << ") -> matchedPatch[" << curPatchElem.elem->getBestMatchIdx() << "]="
                                    << patchMatrix[frameIdx - 1][curPatchElem.elem->getBestMatchIdx()].elem->getIndex()
                                    << std::endl;
                        } else {
                          std::cout << "Orientation " << curPatchElem.elem->getPatchOrientation()
                                    << " selected for unmatched patch " << curPatchElem.elem->getIndex() << " ("
                                    << curPatchElem.elem->getU0() << "," << curPatchElem.elem->getV0() << ")"
                                    << std::endl;
                        }
                      }
                      // if the element is the prediction of a patch in the next
                      // frame, store the current order for the
                      // next frame, to maintain the sequence
                      if ( curPatchElem.nextElemPos >= 0 ) {
                        if ( printDetailedInfo ) {
                          std::cout << "Changing patchMatrixSortedIndexes[" << frameIdx + 1 << "]["
                                    << curPatchElem.nextElemPos << "] ("
                                    << patchMatrixSortedIndexes[frameIdx + 1][curPatchElem.nextElemPos] << ") ->";
                        }
                        patchMatrixSortedIndexes[frameIdx + 1][indNextMatchedPatch++] = curPatchElem.nextElemPos;
                        if ( printDetailedInfo ) {
                          std::cout << "(" << patchMatrixSortedIndexes[frameIdx + 1][curPatchElem.nextElemPos] << ")"
                                    << std::endl;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
          if ( !locationFound ) {
            occupancySizeV *= 2;
            occupancyMap.resize( occupancySizeU * occupancySizeV );
            if ( printDetailedInfo ) {
              std::cout << "Increasing the canvas size (" << occupancySizeU << "," << occupancySizeV << ")"
                        << std::endl;
            }
          } else {
            if ( params_.packingStrategy_ == 2 ) {
              curGlobalElem.getU0()               = best_u;
              curGlobalElem.getV0()               = best_v;
              curGlobalElem.getPatchOrientation() = best_orientation;
              if ( printDetailedInfo ) {
                std::cout << "Selected position (" << best_u << "," << best_v << ") and orientation "
                          << best_orientation << std::endl;
              }
              if ( !( curGlobalElem.isPatchDimensionSwitched() ) ) {
                curPatchElem.elem->getU0() =
                    curGlobalElem.getU0() + ( curPatchElem.elem->getU1() - curGlobalElem.getU1() ) /
                                                curPatchElem.elem->getOccupancyResolution();
                curPatchElem.elem->getV0() =
                    curGlobalElem.getV0() + ( curPatchElem.elem->getV1() - curGlobalElem.getV1() ) /
                                                curPatchElem.elem->getOccupancyResolution();
              } else {
                curPatchElem.elem->getU0() =
                    curGlobalElem.getU0() + ( curPatchElem.elem->getV1() - curGlobalElem.getV1() ) /
                                                curPatchElem.elem->getOccupancyResolution();
                curPatchElem.elem->getV0() =
                    curGlobalElem.getV0() + ( curPatchElem.elem->getU1() - curGlobalElem.getU1() ) /
                                                curPatchElem.elem->getOccupancyResolution();
              }
              curPatchElem.elem->getPatchOrientation() = curGlobalElem.getPatchOrientation();
              sortedPatches.push_back( *curPatchElem.elem );
              if ( printDetailedInfo ) {
                std::cout << "Orientation " << curPatchElem.elem->getPatchOrientation()
                          << " selected for unmatched patch " << curPatchElem.elem->getIndex() << " ("
                          << curPatchElem.elem->getU0() << "," << curPatchElem.elem->getV0() << ")" << std::endl;
              }
              // if the element is the prediction of a patch in the next frame,
              // store the current order for the next
              // frame, to maintain the sequence
              if ( curPatchElem.nextElemPos >= 0 ) {
                if ( printDetailedInfo ) {
                  std::cout << "Changing patchMatrixSortedIndexes[" << frameIdx + 1 << "][" << curPatchElem.nextElemPos
                            << "] (" << patchMatrixSortedIndexes[frameIdx + 1][curPatchElem.nextElemPos] << ") ->";
                }
                patchMatrixSortedIndexes[frameIdx + 1][indNextMatchedPatch++] = curPatchElem.nextElemPos;
                if ( printDetailedInfo ) {
                  std::cout << "(" << patchMatrixSortedIndexes[frameIdx + 1][curPatchElem.nextElemPos] << ")"
                            << std::endl;
                }
              }
              // update the horizon
              curGlobalElem.update_horizon( horizon, top_horizon, bottom_horizon, right_horizon, left_horizon );
              // debugging
              if ( printDetailedInfo ) {
                std::cout << "New Horizon :[";
                for ( int i = 0; i < occupancySizeU; i++ ) { std::cout << horizon[i] << ","; }
                std::cout << "]" << std::endl;
              }
            }
          }
        }
        for ( size_t v0 = 0; v0 < curGlobalElem.getSizeV0(); ++v0 ) {
          for ( size_t u0 = 0; u0 < curGlobalElem.getSizeU0(); ++u0 ) {
            int coord = curGlobalElem.patchBlock2CanvasBlock( u0, v0, occupancySizeU, occupancySizeV );
            if ( params_.lowDelayEncoding_ ) {
              occupancyMap[coord] = true;
            } else {
              occupancyMap[coord] = occupancyMap[coord] || occupancy[v0 * curGlobalElem.getSizeU0() + u0];
            }
          }
        }
        if ( !( curGlobalElem.isPatchDimensionSwitched() ) ) {
          height = ( std::max )(
              height, ( curGlobalElem.getV0() + curGlobalElem.getSizeV0() ) * curGlobalElem.getOccupancyResolution() );
          width = ( std::max )(
              width, ( curGlobalElem.getU0() + curGlobalElem.getSizeU0() ) * curGlobalElem.getOccupancyResolution() );
          maxOccupancyRow = ( std::max )( maxOccupancyRow, ( curGlobalElem.getV0() + curGlobalElem.getSizeV0() ) );
        } else {
          height = ( std::max )(
              height, ( curGlobalElem.getV0() + curGlobalElem.getSizeU0() ) * curGlobalElem.getOccupancyResolution() );
          width = ( std::max )(
              width, ( curGlobalElem.getU0() + curGlobalElem.getSizeV0() ) * curGlobalElem.getOccupancyResolution() );
          maxOccupancyRow = ( std::max )( maxOccupancyRow, ( curGlobalElem.getV0() + curGlobalElem.getSizeU0() ) );
        }

        if ( printDetailedInfo ) {
          if ( params_.packingStrategy_ != 2 ) {
            printMap( occupancyMap, occupancySizeU, occupancySizeV );
          } else {
            printMapTetris( occupancyMap, occupancySizeU, occupancySizeV, horizon );
          }
        }
      }
      // update the sorted list of patches
      frames[frameIdx].getTile( tileIndex ).getPatches() = sortedPatches;

      if ( frames[frameIdx].getTile( tileIndex ).getNumberOfRawPointsPatches() > 0 &&
           !frames[frameIdx].getTile( tileIndex ).getUseRawPointsSeparateVideo() ) {
        packRawPointsPatch( frames[frameIdx].getTile( tileIndex ), occupancyMap, width, height, occupancySizeU,
                            occupancySizeV, maxOccupancyRow );
      }
      if ( params_.enhancedOccupancyMapCode_ &&
           !frames[frameIdx].getTile( tileIndex ).getUseRawPointsSeparateVideo() ) {
        packEOMTexturePointsPatch( frames[frameIdx].getTile( tileIndex ), occupancyMap, width, height, occupancySizeU,
                                   occupancySizeV, maxOccupancyRow );
      }
      if ( printDetailedInfo ) { printMap( occupancyMap, occupancySizeU, occupancySizeV ); }
      std::cout << "actualImageSize " << width << " x " << height << std::endl;
    }
  }
}

void PCCEncoder::packFlexible( PCCFrameContext& tile,
                               int              packingStrategy,
                               size_t           presetWidth,
                               size_t           presetHeight,
                               int              safeguard,
                               bool             enablePointCloudPartitioning ) {
  auto  width   = tile.getWidth();
  auto& height  = tile.getHeight();
  auto& patches = tile.getPatches();
  // set no matched patches, since this function does not take into account the
  // previous frame
  tile.setNumMatchedPatches( 0 );
  if ( patches.empty() ) {
    if ( tile.getNumberOfRawPointsPatches() == 0 ) { return; }
    if ( tile.getUseRawPointsSeparateVideo() ) { return; }
    std::vector<bool> occupancyMap;
    size_t            occupancySizeU = presetWidth / params_.occupancyResolution_;
    size_t            occupancySizeV = presetHeight / params_.occupancyResolution_;
    if ( presetWidth == 0 || presetHeight == 0 ) {
      auto& rawPointsPatch       = tile.getRawPointsPatch( 0 );
      auto  rawPointsPatchBlocks = static_cast<size_t>(
          ceil( double( rawPointsPatch.sizeX() ) / ( params_.occupancyResolution_ * params_.occupancyResolution_ ) ) );
      if ( presetWidth == 0 ) occupancySizeU = params_.minimumImageWidth_ / params_.occupancyResolution_;
      if ( presetHeight == 0 )
        occupancySizeV = static_cast<size_t>( ceil( double( rawPointsPatchBlocks ) / occupancySizeU ) );
    }
    occupancyMap.resize( occupancySizeU * occupancySizeV );
    if ( tile.getNumberOfRawPointsPatches() > 0 && !tile.getUseRawPointsSeparateVideo() ) {
      packRawPointsPatch( tile, occupancyMap, width, height, occupancySizeU, occupancySizeV, 0 );
    } else {
      if ( printDetailedInfo ) { printMap( occupancyMap, occupancySizeU, occupancySizeV ); }
    }
    if ( params_.enhancedOccupancyMapCode_ && !tile.getUseRawPointsSeparateVideo() ) {
      packEOMTexturePointsPatch( tile, occupancyMap, width, height, occupancySizeU, occupancySizeV, 0 );
    }
    if ( printDetailedInfo ) { printMap( occupancyMap, occupancySizeU, occupancySizeV ); }
    bool emptyTile = false;
    if ( height == 0 ) {
      emptyTile = true;
      height    = 64;
    }
    std::cout << "frame " << tile.getFrameIndex() << " tile " << tile.getTileIndex()
              << " packFlexible(patchEmpty): actualImageSize " << width << " x " << height;
    if ( emptyTile )
      std::cout << " height adjusted" << std::endl;
    else
      std::cout << std::endl;
    return;
  }
  // sorting by patch largest dimension
  if ( packingStrategy == 0 ) {
    std::sort( patches.begin(), patches.end() );
  } else {
    std::sort( patches.begin(), patches.end(), []( PCCPatch& a, PCCPatch& b ) { return a.gt( b ); } );
  }
  if ( printDetailedInfo ) {
    std::cout << "Patch order:" << std::endl;
    for ( auto& patch : patches ) {
      std::cout << "Patch[" << patch.getIndex() << "]=(" << patch.getSizeU0() << "," << patch.getSizeV0() << ")"
                << std::endl;
    }
  }
  size_t occupancySizeU = presetWidth / params_.occupancyResolution_;
  size_t occupancySizeV = ( std::max )( patches[0].getSizeV0(), patches[0].getSizeU0() );
  for ( auto& patch : patches ) { occupancySizeU = ( std::max )( occupancySizeU, patch.getSizeU0() + 1 ); }

  int numROIs     = params_.numROIs_;
  int numTilesHor = params_.numTilesHor_;
  int tileWidth   = occupancySizeU / numTilesHor;
  int tileHeight  = int( tileWidth * params_.tileHeightToWidthRatio_ );
  if ( params_.enablePointCloudPartitioning_ )
    std::cout << "frame " << tile.getFrameIndex() << " tilesize: " << tileWidth << "x" << tileHeight << std::endl;
  occupancySizeV = ( occupancySizeV >= tileHeight ) ? occupancySizeV : tileHeight;

  // width  = occupancySizeU * params_.occupancyResolution_;
  height = occupancySizeV * params_.occupancyResolution_;
  size_t maxOccupancyRow{0};

  std::vector<bool> occupancyMap;
  int               numOrientations = ( packingStrategy == 0 ) ? 1 : ( params_.useEightOrientations_ ? 8 : 2 );
  occupancyMap.resize( occupancySizeU * occupancySizeV, false );
  for ( auto& patch : patches ) {
    assert( patch.getSizeU0() <= occupancySizeU );
    assert( patch.getSizeV0() <= occupancySizeV );
    bool  locationFound = false;
    auto& occupancy     = patch.getOccupancy();
    while ( !locationFound ) {
      for ( size_t v = 0; v < occupancySizeV && !locationFound; ++v ) {
        for ( size_t u = 0; u < occupancySizeU && !locationFound; ++u ) {
          patch.getU0() = u;
          patch.getV0() = v;
          for ( size_t orientationIdx = 0; orientationIdx < numOrientations && !locationFound; orientationIdx++ ) {
            if ( packingStrategy == 0 )
              patch.getPatchOrientation() = PATCH_ORIENTATION_DEFAULT;
            else {
              if ( patch.getSizeU0() > patch.getSizeV0() ) {
                patch.getPatchOrientation() = orientation_horizontal[orientationIdx];
              } else {
                patch.getPatchOrientation() = orientation_vertical[orientationIdx];
              }
            }
            if ( patch.checkFitPatchCanvas( occupancyMap, occupancySizeU, occupancySizeV, params_.lowDelayEncoding_,
                                            safeguard ) ) {
              locationFound = true;
              if ( printDetailedInfo ) {
                std::cout << "Orientation " << patch.getPatchOrientation() << " selected for patch " << patch.getIndex()
                          << " (" << u << "," << v << ")" << std::endl;
              }
            }
          }
        }
      }
      if ( !locationFound ) {
        occupancySizeV *= 2;
        occupancyMap.resize( occupancySizeU * occupancySizeV );
      }
    }
    for ( size_t v0 = 0; v0 < patch.getSizeV0(); ++v0 ) {
      for ( size_t u0 = 0; u0 < patch.getSizeU0(); ++u0 ) {
        int coord = patch.patchBlock2CanvasBlock( u0, v0, occupancySizeU, occupancySizeV );
        if ( params_.lowDelayEncoding_ ) {
          occupancyMap[coord] = true;
        } else {
          occupancyMap[coord] = occupancyMap[coord] || occupancy[v0 * patch.getSizeU0() + u0];
        }
      }
    }

    if ( !( patch.isPatchDimensionSwitched() ) ) {
      height = ( std::max )( height, ( patch.getV0() + patch.getSizeV0() ) * patch.getOccupancyResolution() );
      // width  = ( std::max )( width, ( patch.getU0() + patch.getSizeU0() ) * patch.getOccupancyResolution() );
      maxOccupancyRow = ( std::max )( maxOccupancyRow, ( patch.getV0() + patch.getSizeV0() ) );
    } else {
      height = ( std::max )( height, ( patch.getV0() + patch.getSizeU0() ) * patch.getOccupancyResolution() );
      // width  = ( std::max )( width, ( patch.getU0() + patch.getSizeV0() ) * patch.getOccupancyResolution() );
      maxOccupancyRow = ( std::max )( maxOccupancyRow, ( patch.getV0() + patch.getSizeU0() ) );
    }
  }

  if ( tile.getNumberOfRawPointsPatches() > 0 && !tile.getUseRawPointsSeparateVideo() ) {
    packRawPointsPatch( tile, occupancyMap, width, height, occupancySizeU, occupancySizeV, maxOccupancyRow );
  } else {
    if ( printDetailedInfo ) { printMap( occupancyMap, occupancySizeU, occupancySizeV ); }
  }
  if ( params_.enhancedOccupancyMapCode_ && !tile.getUseRawPointsSeparateVideo() ) {
    packEOMTexturePointsPatch( tile, occupancyMap, width, height, occupancySizeU, occupancySizeV, maxOccupancyRow );
  }
  if ( printDetailedInfo ) { printMap( occupancyMap, occupancySizeU, occupancySizeV ); }
  std::cout << "frame " << tile.getFrameIndex() << " tile " << tile.getTileIndex() << " packFlexible: actualImageSize "
            << width << " x " << height << std::endl;
}

void PCCEncoder::packMultipleTiles( PCCAtlasFrameContext& atlasFrame, int safeguard ) {
  auto& frame   = atlasFrame.getTitleFrameContext();
  auto& width   = frame.getWidth();
  auto& height  = frame.getHeight();
  auto& patches = frame.getPatches();
  if ( patches.empty() ) { return; }
  std::sort( patches.begin(), patches.end() );
  size_t occupancySizeU = params_.minimumImageWidth_ / params_.occupancyResolution_;
  size_t occupancySizeV =
      ( std::max )( params_.minimumImageHeight_ / params_.occupancyResolution_, patches[0].getSizeV0() );
  for ( auto& patch : patches ) { occupancySizeU = ( std::max )( occupancySizeU, patch.getSizeU0() + 1 ); }

  int numROIs     = params_.numROIs_;
  int numTilesHor = params_.numTilesHor_;
  // should we force the tile size to be a multiple of 64/params_.occupancyResolution_???
  // Can we have tiles with smaller blocks? Because we may have that case for occupancy!!!!
  int tileWidth  = occupancySizeU / numTilesHor;
  int tileHeight = int( tileWidth * params_.tileHeightToWidthRatio_ );
  if ( params_.enablePointCloudPartitioning_ )
    std::cout << "frame " << frame.getFrameIndex() << " tilesize: " << tileWidth << "x" << tileHeight << std::endl;
  occupancySizeV = ( occupancySizeV >= tileHeight ) ? occupancySizeV : tileHeight;

  width  = occupancySizeU * params_.occupancyResolution_;
  height = occupancySizeV * params_.occupancyResolution_;
  size_t maxOccupancyRow{0};

  std::vector<bool> occupancyMap;
  occupancyMap.resize( occupancySizeU * occupancySizeV, false );

  std::vector<Tile> tilesNotAvailable;  // set of all tiles occupied by prev
                                        // ROIs of current ROI
  int lastOccupiedTileIndex          = -1;
  int lastOccupiedTileIndexByPrevROI = -1;
  // loop over ROIs
  for ( int roiIndex = 0; roiIndex < numROIs; ++roiIndex ) {
    tilesNotAvailable.clear();
    if ( roiIndex > 0 ) {
      lastOccupiedTileIndexByPrevROI = lastOccupiedTileIndex;
      for ( int tileIndex = 0; tileIndex <= lastOccupiedTileIndexByPrevROI; ++tileIndex ) {
        Tile tile;
        tile.minU = ( tileIndex % numTilesHor ) * tileWidth;
        tile.minV = ( tileIndex / numTilesHor ) * tileHeight;
        tile.maxU = tile.minU + tileWidth - 1;
        tile.maxV = tile.minV + tileHeight - 1;
        tilesNotAvailable.push_back( tile );
      }
    }
    // loop over patches of current ROI
    for ( auto& patch : patches ) {
      if ( roiIndex != patch.getRoiIndex() ) { continue; }
      assert( patch.getSizeU0() <= occupancySizeU );
      assert( patch.getSizeV0() <= occupancySizeV );
      bool  locationFound = false;
      auto& occupancy     = patch.getOccupancy();
      while ( !locationFound ) {
        patch.getPatchOrientation() = PATCH_ORIENTATION_DEFAULT;  // only allowed orientation in anchor
        int numTilesAvailable       = ceil( double( occupancySizeV ) / double( tileHeight ) ) * numTilesHor;
        for ( int tileIndex = lastOccupiedTileIndexByPrevROI + 1; tileIndex < numTilesAvailable && !locationFound;
              ++tileIndex ) {
          Tile tile;
          tile.minU = ( tileIndex % numTilesHor ) * tileWidth;
          tile.maxU = tile.minU + tileWidth - 1;
          tile.minV = ( tileIndex / numTilesHor ) * tileHeight;
          tile.maxV = tile.minV + tileHeight - 1;
          for ( int v = 0; v <= occupancySizeV && !locationFound; ++v ) {
            for ( int u = 0; u <= occupancySizeU && !locationFound; ++u ) {
              patch.getU0()        = u;
              patch.getV0()        = v;
              bool tileIsAvailable = true;
              if ( roiIndex > 0 ) {
                for ( auto& tile : tilesNotAvailable ) {
                  if ( ( tile.minU <= u && u <= tile.maxU ) && ( tile.minV <= v && v <= tile.maxV ) &&
                       ( !locationFound ) ) {
                    tileIsAvailable = false;
                    break;
                  }
                }
              }
              if ( tileIsAvailable ) {
                if ( patch.checkFitPatchCanvas( occupancyMap, occupancySizeU, occupancySizeV, params_.lowDelayEncoding_,
                                                safeguard, tile ) ) {
                  locationFound = true;
                  if ( tileIndex > lastOccupiedTileIndex ) { lastOccupiedTileIndex = tileIndex; }
                  std::cout << "intra: ROI[" << roiIndex << "] patch " << patch.getIndex() << "\t@(" << patch.getU0()
                            << "," << patch.getV0() << ")\ts(" << patch.getSizeU0() << "x" << patch.getSizeV0()
                            << ")\to" << patch.getPatchOrientation() << " fitted in tile-" << tileIndex + 1 << "/"
                            << numTilesAvailable << "-----[" << tile.minU << "," << tile.maxU << "][" << tile.minV
                            << "," << tile.maxV << "]" << std::endl;
                }
              }
            }
          }
        }
        if ( !locationFound ) {
          occupancySizeV *= 2;
          occupancyMap.resize( occupancySizeU * occupancySizeV );
        }
      }
      for ( size_t v0 = 0; v0 < patch.getSizeV0(); ++v0 ) {
        const size_t v = patch.getV0() + v0;
        for ( size_t u0 = 0; u0 < patch.getSizeU0(); ++u0 ) {
          const size_t u = patch.getU0() + u0;
          occupancyMap[v * occupancySizeU + u] =
              occupancyMap[v * occupancySizeU + u] || occupancy[v0 * patch.getSizeU0() + u0];
        }
      }
      height          = ( std::max )( height, ( patch.getV0() + patch.getSizeV0() ) * patch.getOccupancyResolution() );
      width           = ( std::max )( width, ( patch.getU0() + patch.getSizeU0() ) * patch.getOccupancyResolution() );
      maxOccupancyRow = ( std::max )( maxOccupancyRow, ( patch.getV0() + patch.getSizeV0() ) );
    }  // patch loop
    printMap( occupancyMap, occupancySizeU, occupancySizeV );
  }  // ROI loop
  std::cout << "frame " << atlasFrame.getAtlasFrameIndex() << " packMultipleTiles: actualImageSize " << width << " x "
            << height << std::endl;
}
void PCCEncoder::packFlexibleMultipleTiles( PCCAtlasFrameContext& atlasFrame, int safeguard ) {
  auto& frame              = atlasFrame.getTitleFrameContext();
  auto& partitionToTileMap = atlasFrame.getPartitionToTileMap();
  auto& width              = frame.getWidth();
  auto& height             = frame.getHeight();
  auto& patches            = frame.getPatches();

  // set no matched patches, since this function does not take into account the previous frame
  frame.setNumMatchedPatches( 0 );
  if ( patches.empty() ) { return; }
  // sorting by patch largest dimension
  std::sort( patches.begin(), patches.end(), []( PCCPatch& a, PCCPatch& b ) { return a.gt( b ); } );
  if ( printDetailedInfo ) {
    std::cout << "Patch order:" << std::endl;
    for ( auto& patch : patches ) {
      std::cout << "Patch[" << patch.getIndex() << "]=(" << patch.getSizeU0() << "," << patch.getSizeV0() << ")"
                << std::endl;
    }
  }

  size_t occupancySizeU = params_.minimumImageWidth_ / params_.occupancyResolution_;
  size_t occupancySizeV = ( std::max )( patches[0].getSizeV0(), patches[0].getSizeU0() );
  for ( auto& patch : patches ) { occupancySizeU = ( std::max )( occupancySizeU, patch.getSizeU0() + 1 ); }

  int numROIs     = params_.numROIs_;
  int numTilesHor = params_.numTilesHor_;
  int tileWidth   = occupancySizeU / numTilesHor;
  int tileHeight  = int( tileWidth * params_.tileHeightToWidthRatio_ );
  if ( params_.enablePointCloudPartitioning_ )
    std::cout << "frame " << frame.getFrameIndex() << " tilesize: " << tileWidth << "x" << tileHeight << std::endl;
  occupancySizeV  = ( occupancySizeV >= tileHeight ) ? occupancySizeV : tileHeight;
  int numTilesVer = occupancySizeV / tileHeight;
  // initializating the tile map to -1 (not assigned)
  partitionToTileMap.resize( numTilesHor * numTilesVer, -1 );

  width  = occupancySizeU * params_.occupancyResolution_;
  height = occupancySizeV * params_.occupancyResolution_;
  size_t maxOccupancyRow{0};

  std::vector<bool> occupancyMap;
  vector<int>       orientation_vertical = {
      PATCH_ORIENTATION_DEFAULT, PATCH_ORIENTATION_SWAP,    PATCH_ORIENTATION_ROT180,
      PATCH_ORIENTATION_MIRROR,  PATCH_ORIENTATION_MROT180, PATCH_ORIENTATION_ROT270,
      PATCH_ORIENTATION_MROT90,  PATCH_ORIENTATION_ROT90};  // favoring vertical orientation
  vector<int> orientation_horizontal = {
      PATCH_ORIENTATION_SWAP,   PATCH_ORIENTATION_DEFAULT, PATCH_ORIENTATION_ROT270,
      PATCH_ORIENTATION_MROT90, PATCH_ORIENTATION_ROT90,   PATCH_ORIENTATION_ROT180,
      PATCH_ORIENTATION_MIRROR, PATCH_ORIENTATION_MROT180};  // favoring horizontal orientations (that should be
                                                             // rotated)
  int numOrientations = params_.useEightOrientations_ ? 8 : 2;
  occupancyMap.resize( occupancySizeU * occupancySizeV, false );
  std::vector<Tile> tilesNotAvailable;  // set of all tiles occupied by prev
                                        // ROIs of current ROI
  int lastOccupiedTileIndex          = -1;
  int lastOccupiedTileIndexByPrevROI = -1;
  // loop over ROIs
  for ( int roiIndex = 0; roiIndex < numROIs; ++roiIndex ) {
    // calculate the position which the tile group will start: top left available tile
    int  tileStartPosU = -1;
    int  tileStartPosV = -1;
    int  tileEndPosU   = -1;
    bool foundLimits   = false;
    while ( !foundLimits ) {
      for ( int v = 0; ( v < ceil( double( occupancySizeV ) / double( tileHeight ) ) ) && !foundLimits; v++ ) {
        for ( int u = 0; ( u < numTilesHor ) && !foundLimits; u++ ) {
          bool tileIsAvailable = ( partitionToTileMap[u + numTilesHor * v] == -1 );
          if ( tileIsAvailable ) {
            tileStartPosU = u;
            tileStartPosV = v;
            // found the top left corner, now keep going to the right to find the end position
            tileEndPosU              = tileStartPosU;
            bool nextTileIsAvailable = true;
            do {
              // check if the next tiles in the horizontal line are available
              if ( tileEndPosU + 1 < numTilesHor ) {
                nextTileIsAvailable = ( partitionToTileMap[( tileEndPosU + 1 ) + numTilesHor * v] == -1 );
                if ( nextTileIsAvailable )
                  tileEndPosU++;
                else
                  nextTileIsAvailable = false;
              } else
                nextTileIsAvailable = false;
            } while ( nextTileIsAvailable );
            foundLimits = true;
          }
        }
      }
      if ( !foundLimits ) {
        // the map might not have any available tiles, let's increase the size of the canvas
        occupancySizeV *= 2;
        occupancyMap.resize( occupancySizeU * occupancySizeV );
        // resizing the partitionToTile as well
        int numTilesVerNew = occupancySizeV / tileHeight;
        partitionToTileMap.resize( numTilesHor * numTilesVerNew );
        for ( int i = numTilesHor * numTilesVer; i < numTilesHor * numTilesVerNew; i++ ) partitionToTileMap[i] = -1;
        numTilesVer = numTilesVerNew;
      }
    }
    // loop over patches of current ROI
    for ( auto& patch : patches ) {
      if ( roiIndex != patch.getRoiIndex() ) { continue; }
      assert( patch.getSizeU0() <= occupancySizeU );
      assert( patch.getSizeV0() <= occupancySizeV );
      bool  locationFound = false;
      auto& occupancy     = patch.getOccupancy();
      // fit patch in available tiles (i.e., tiles not occupied by previous
      // ROIs)
      while ( !locationFound ) {
        // loop on the size fo the tile group, to fit the patch in the smallest area possible first
        for ( size_t numTilesInTileVert = 1; numTilesInTileVert <= numTilesVer && !locationFound;
              numTilesInTileVert++ ) {
          for ( size_t numTilesInTileHor = 1;
                numTilesInTileHor <= ( tileEndPosU - tileStartPosU + 1 ) && !locationFound; numTilesInTileHor++ ) {
            Tile tile;
            tile.minU = (tileStartPosU)*tileWidth;
            tile.maxU = tile.minU + tileWidth * (numTilesInTileHor)-1;
            tile.minV = (tileStartPosV)*tileHeight;
            tile.maxV = tile.minV + tileHeight * (numTilesInTileVert)-1;
            // now look for a possible position in the defined tile group
            for ( size_t v = 0; v < occupancySizeV && !locationFound; ++v ) {
              for ( size_t u = 0; u < occupancySizeU && !locationFound; ++u ) {
                patch.getU0() = u;
                patch.getV0() = v;
                for ( size_t orientationIdx = 0; orientationIdx < numOrientations && !locationFound;
                      orientationIdx++ ) {
                  if ( patch.getSizeU0() > patch.getSizeV0() ) {
                    patch.getPatchOrientation() = orientation_horizontal[orientationIdx];
                  } else {
                    patch.getPatchOrientation() = orientation_vertical[orientationIdx];
                  }
                  if ( patch.checkFitPatchCanvas( occupancyMap, occupancySizeU, occupancySizeV,
                                                  params_.lowDelayEncoding_, safeguard, tile ) ) {
                    locationFound = true;
                    std::cout << "intra: ROI[" << roiIndex << "] patch " << patch.getIndex() << "\t@(" << patch.getU0()
                              << "," << patch.getV0() << ")\ts(" << patch.getSizeU0() << "x" << patch.getSizeV0()
                              << ")\to" << patch.getPatchOrientation()
                              << " fitted in tile-"
                                 "["
                              << tile.minU << "," << tile.maxU << "][" << tile.minV << "," << tile.maxV << "]"
                              << std::endl;
                    if ( printDetailedInfo ) {
                      std::cout << "Orientation " << patch.getPatchOrientation() << " selected for patch "
                                << patch.getIndex() << " (" << u << "," << v << ")" << std::endl;
                    }
                  }
                }
              }
            }
          }
        }
        if ( !locationFound ) {
          occupancySizeV *= 2;
          occupancyMap.resize( occupancySizeU * occupancySizeV );
          // resizing the partitionToTile as well
          int numTilesVerNew = occupancySizeV / tileHeight;
          partitionToTileMap.resize( numTilesHor * numTilesVerNew );
          for ( int i = numTilesHor * numTilesVer; i < numTilesHor * numTilesVerNew; i++ ) partitionToTileMap[i] = -1;
          numTilesVer = numTilesVerNew;
        }
      }  // while loop
      for ( size_t v0 = 0; v0 < patch.getSizeV0(); ++v0 ) {
        for ( size_t u0 = 0; u0 < patch.getSizeU0(); ++u0 ) {
          int coord = patch.patchBlock2CanvasBlock( u0, v0, occupancySizeU, occupancySizeV );
          if ( params_.lowDelayEncoding_ )
            occupancyMap[coord] = true;
          else
            occupancyMap[coord] = occupancyMap[coord] || occupancy[v0 * patch.getSizeU0() + u0];
          // also claim the tile for the ROI
          size_t x, y;
          patch.patch2Canvas( u0, v0, occupancySizeU * patch.getOccupancyResolution(),
                              occupancySizeV * patch.getOccupancyResolution(), x, y );
          int tileIndex = ( x / patch.getOccupancyResolution() ) / tileWidth +
                          numTilesHor * ( ( y / patch.getOccupancyResolution() ) / tileHeight );
          if ( tileIndex >= partitionToTileMap.size() )
            std::cout << "ERROR" << std::endl;
          else
            partitionToTileMap[tileIndex] = roiIndex;
        }
      }

      if ( !( patch.isPatchDimensionSwitched() ) ) {
        height = ( std::max )( height, ( patch.getV0() + patch.getSizeV0() ) * patch.getOccupancyResolution() );
        width  = ( std::max )( width, ( patch.getU0() + patch.getSizeU0() ) * patch.getOccupancyResolution() );
        maxOccupancyRow = ( std::max )( maxOccupancyRow, ( patch.getV0() + patch.getSizeV0() ) );
      } else {
        height = ( std::max )( height, ( patch.getV0() + patch.getSizeU0() ) * patch.getOccupancyResolution() );
        width  = ( std::max )( width, ( patch.getU0() + patch.getSizeV0() ) * patch.getOccupancyResolution() );
        maxOccupancyRow = ( std::max )( maxOccupancyRow, ( patch.getV0() + patch.getSizeU0() ) );
      }
    }  // patch loop
    if ( printDetailedInfo ) { printMap( occupancyMap, occupancySizeU, occupancySizeV ); }
  }  // ROI loop
  std::cout << "frame " << atlasFrame.getAtlasFrameIndex() << " packFlexibleMultipleTiles: actualImageSize " << width
            << " x " << height << std::endl;
}
void PCCEncoder::spatialConsistencyPackMultipleTiles( PCCAtlasFrameContext& atlasFrame,
                                                      PCCAtlasFrameContext& prevAtlasFrame,
                                                      int                   safeguard ) {
  auto& frame       = atlasFrame.getTitleFrameContext();
  auto& prevFrame   = prevAtlasFrame.getTitleFrameContext();
  auto& width       = frame.getWidth();
  auto& height      = frame.getHeight();
  auto& patches     = frame.getPatches();
  auto& prevPatches = prevFrame.getPatches();
  if ( patches.empty() ) { return; }
  std::sort( patches.begin(), patches.end() );
  int    id             = 0;
  size_t occupancySizeU = params_.minimumImageWidth_ / params_.occupancyResolution_;
  size_t occupancySizeV =
      ( std::max )( params_.minimumImageHeight_ / params_.occupancyResolution_, patches[0].getSizeV0() );
  vector<PCCPatch> matchedPatches, tmpPatches, newOrderPatches;
  matchedPatches.clear();
  newOrderPatches.clear();
  float thresholdIOU = 0.2f;

  // main loop. (NOTICE: enforcing the match to be from the same ROI)
  for ( auto& patch : prevPatches ) {
    assert( patch.getSizeU0() <= occupancySizeU );
    assert( patch.getSizeV0() <= occupancySizeV );
    id++;
    float maxIou  = 0.0f;
    int   bestIdx = -1, cId = 0;
    for ( auto& cpatch : patches ) {
      if ( ( patch.getViewId() == cpatch.getViewId() ) && ( cpatch.getBestMatchIdx() == InvalidPatchIndex ) &&
           ( patch.getLodScaleX() == cpatch.getLodScaleX() && patch.getLodScaleY() == cpatch.getLodScaleY() ) &&
           ( patch.getRoiIndex() == cpatch.getRoiIndex() ) ) {
        Rect  rect  = Rect( patch.getU1(), patch.getV1(), patch.getSizeU(), patch.getSizeV() );
        Rect  crect = Rect( cpatch.getU1(), cpatch.getV1(), cpatch.getSizeU(), cpatch.getSizeV() );
        float iou   = computeIOU( rect, crect );
        if ( iou > maxIou ) {
          maxIou  = iou;
          bestIdx = cId;
        }
      }  // end of if (patch.viewId == cpatch.viewId).
      cId++;
    }
    if ( maxIou > thresholdIOU ) {
      // store the best match index
      patches[bestIdx].setBestMatchIdx( id - 1 );  // the matched patch id in previous frame.
      patches[bestIdx].setPatchType( (uint8_t)P_INTER );
      matchedPatches.push_back( patches[bestIdx] );
    }
  }

  // generate new patch order.
  newOrderPatches = matchedPatches;
  for ( auto patch : patches ) {
    assert( patch.getSizeU0() <= occupancySizeU );
    assert( patch.getSizeV0() <= occupancySizeV );
    if ( patch.getBestMatchIdx() == InvalidPatchIndex ) {
      patch.setPatchType( (uint8_t)P_INTRA );  // P_TYPE_INTRA
      newOrderPatches.push_back( patch );
    }
  }

  frame.setNumMatchedPatches( matchedPatches.size() );
  // remove the below logs when useless.
  patches = newOrderPatches;
  for ( auto& patch : patches ) { occupancySizeU = ( std::max )( occupancySizeU, patch.getSizeU0() + 1 ); }

  int numTilesHor = params_.numTilesHor_;
  int tileWidth   = occupancySizeU / numTilesHor;
  int tileHeight  = int( tileWidth * params_.tileHeightToWidthRatio_ );
  if ( params_.enablePointCloudPartitioning_ )
    std::cout << "frame " << frame.getFrameIndex() << " tilesize: " << tileWidth << "x" << tileHeight << std::endl;
  occupancySizeV = ( occupancySizeV >= tileHeight ) ? occupancySizeV : tileHeight;

  width  = occupancySizeU * params_.occupancyResolution_;
  height = occupancySizeV * params_.occupancyResolution_;
  size_t maxOccupancyRow{0};

  std::vector<bool> occupancyMap;
  occupancyMap.resize( occupancySizeU * occupancySizeV, false );
  std::vector<Tile> tilesNotAvailable;
  int               numROIs                        = params_.numROIs_;
  int               lastOccupiedTileIndex          = -1;
  int               lastOccupiedTileIndexHor       = -1;
  int               lastOccupiedTileIndexVer       = -1;
  int               lastOccupiedTileIndexByPrevROI = -1;
  int               numTilesAvailable;
  // loop over ROIs
  for ( size_t roiIndex = 0; roiIndex < numROIs; ++roiIndex ) {
    // find top left corner to start placing the tile group, and determine the maximum horizontal size
    int numTilesVert = ceil( double( occupancySizeV ) / double( tileHeight ) );
    for ( int y = 0; y < numTilesVert; y++ )
      for ( int x = 0; x < numTilesHor; x++ )

        // loop over patches of current ROI
        for ( auto& patch : patches ) {
          if ( roiIndex != patch.getRoiIndex() ) { continue; }
          assert( patch.getSizeU0() <= occupancySizeU );
          assert( patch.getSizeV0() <= occupancySizeV );
          bool  locationFound = false;
          auto& occupancy     = patch.getOccupancy();
          while ( !locationFound ) {
            patch.getPatchOrientation() = PATCH_ORIENTATION_DEFAULT;  // only one orientation is allowed
            numTilesAvailable           = ceil( double( occupancySizeV ) / double( tileHeight ) ) * numTilesHor;
            for ( int tileIndex = lastOccupiedTileIndexByPrevROI + 1; tileIndex < numTilesAvailable && !locationFound;
                  ++tileIndex ) {
              Tile tile;
              tile.minU = ( tileIndex % numTilesHor ) * tileWidth;
              tile.maxU = tile.minU + tileWidth - 1;
              tile.minV = ( tileIndex / numTilesHor ) * tileHeight;
              tile.maxV = tile.minV + tileHeight - 1;
              for ( size_t v = 0; v <= occupancySizeV && !locationFound; ++v ) {
                for ( size_t u = 0; u <= occupancySizeU && !locationFound; ++u ) {
                  patch.getU0()        = u;
                  patch.getV0()        = v;
                  bool tileIsAvailable = true;
                  if ( roiIndex > 0 ) {
                    for ( auto& tile : tilesNotAvailable ) {
                      if ( ( tile.minU <= u && u <= tile.maxU ) && ( tile.minV <= v && v <= tile.maxV ) &&
                           ( !locationFound ) ) {
                        tileIsAvailable = false;
                        break;
                      }
                    }
                  }
                  if ( tileIsAvailable ) {
                    if ( patch.checkFitPatchCanvas( occupancyMap, occupancySizeU, occupancySizeV,
                                                    params_.lowDelayEncoding_, safeguard, tile ) ) {
                      locationFound = true;
                      if ( tileIndex > lastOccupiedTileIndex ) { lastOccupiedTileIndex = tileIndex; }
                      std::cout << "ROI[" << roiIndex << "] patch " << patch.getIndex() << "\t@(" << patch.getU0()
                                << "," << patch.getV0() << ")\ts(" << patch.getSizeU0() << "x" << patch.getSizeV0()
                                << ")\to" << patch.getPatchOrientation() << " fitted in tile-" << tileIndex + 1 << "/"
                                << numTilesAvailable << "-----[" << tile.minU << "," << tile.maxU << "][" << tile.minV
                                << "," << tile.maxV << "]"
                                << " +-+-+-+-+-(NOT MATCHED patch)-+-+-+-+-+" << std::endl;
                    }
                  }
                }
              }
            }
            if ( !locationFound ) {
              occupancySizeV *= 2;
              occupancyMap.resize( occupancySizeU * occupancySizeV );
            }
          }
          for ( size_t v0 = 0; v0 < patch.getSizeV0(); ++v0 ) {
            const size_t v = patch.getV0() + v0;
            for ( size_t u0 = 0; u0 < patch.getSizeU0(); ++u0 ) {
              const size_t u = patch.getU0() + u0;
              occupancyMap[v * occupancySizeU + u] =
                  occupancyMap[v * occupancySizeU + u] || occupancy[v0 * patch.getSizeU0() + u0];
            }
          }

          height = ( std::max )( height, ( patch.getV0() + patch.getSizeV0() ) * patch.getOccupancyResolution() );
          width  = ( std::max )( width, ( patch.getU0() + patch.getSizeU0() ) * patch.getOccupancyResolution() );
          maxOccupancyRow = ( std::max )( maxOccupancyRow, ( patch.getV0() + patch.getSizeV0() ) );
          // print(occupancyMap, occupancySizeU, occupancySizeV);
        }  // patch loop
    printMap( occupancyMap, occupancySizeU, occupancySizeV );
  }  // ROI loop
  std::cout << "frame " << atlasFrame.getAtlasFrameIndex() << " spatialConsistencypackMultipleTiles: actualImageSize "
            << width << " x" << height << std::endl;
}

void PCCEncoder::spatialConsistencyPackFlexibleMultipleTiles( PCCAtlasFrameContext& atlasFrame,
                                                              PCCAtlasFrameContext& prevAtlasFrame,
                                                              int                   safeguard ) {
  auto& frame              = atlasFrame.getTitleFrameContext();
  auto& prevFrame          = prevAtlasFrame.getTitleFrameContext();
  auto& width              = frame.getWidth();
  auto& height             = frame.getHeight();
  auto& patches            = frame.getPatches();
  auto& partitionToTileMap = atlasFrame.getPartitionToTileMap();

  auto& prevPatches = prevFrame.getPatches();
  if ( patches.empty() ) { return; }
  std::sort( patches.begin(), patches.end(), []( PCCPatch& a, PCCPatch& b ) { return a.gt( b ); } );
  int              id             = 0;
  size_t           occupancySizeU = params_.minimumImageWidth_ / params_.occupancyResolution_;
  size_t           occupancySizeV = ( std::max )( patches[0].getSizeU0(), patches[0].getSizeV0() );
  vector<PCCPatch> matchedPatches, tmpPatches;
  matchedPatches.clear();
  float  thresholdIOU    = 0.2f;
  size_t bestRefFrameIdx = 0;
  // main loop. (NOTE: enforcing the matches to be from the same ROI)
  for ( auto& patch : prevPatches ) {
    id++;
    float maxIou  = 0.0f;
    int   bestIdx = -1, cId = 0;
    for ( auto& cpatch : patches ) {
      if ( ( patch.getViewId() == cpatch.getViewId() ) && ( cpatch.getBestMatchIdx() == InvalidPatchIndex ) &&
           ( patch.getLodScaleX() == cpatch.getLodScaleX() && patch.getLodScaleY() == cpatch.getLodScaleY() &&
             ( patch.getRoiIndex() == cpatch.getRoiIndex() ) ) ) {
        patch.setPatchType( (uint8_t)P_INTRA );
        Rect  rect  = Rect( patch.getU1(), patch.getV1(), patch.getSizeU(), patch.getSizeV() );
        Rect  crect = Rect( cpatch.getU1(), cpatch.getV1(), cpatch.getSizeU(), cpatch.getSizeV() );
        float iou   = computeIOU( rect, crect );
        if ( iou > maxIou ) {
          maxIou  = iou;
          bestIdx = cId;
        }
      }  // end of if (patch.viewId == cpatch.viewId).
      cId++;
    }
    if ( maxIou > thresholdIOU ) {
      // store the best match index
      patches[bestIdx].setBestMatchIdx( id - 1 );  // the matched patch id in preivious frame.
      patches[bestIdx].setPatchType( (uint8_t)P_INTER );
      patches[bestIdx].setRefAtlasFrameIndex( bestRefFrameIdx );
      matchedPatches.push_back( patches[bestIdx] );
    }
  }
  // generate new patch order.
  vector<PCCPatch> newOrderPatches = matchedPatches;
  for ( auto patch : patches ) {
    assert( patch.getSizeU0() <= occupancySizeU );
    assert( patch.getSizeV0() <= occupancySizeV );
    if ( patch.getBestMatchIdx() == InvalidPatchIndex ) {
      patch.setPatchType( (uint8_t)P_INTRA );
      newOrderPatches.push_back( patch );
    }
  }
  frame.setNumMatchedPatches( matchedPatches.size() );
  // remove the below logs when useless.
  if ( printDetailedInfo ) {
    std::cout << "patches.size:" << patches.size() << ",reOrderedPatches.size:" << newOrderPatches.size()
              << ",matchedpatches.size:" << frame.getNumMatchedPatches() << std::endl;
  }
  patches = newOrderPatches;
  if ( printDetailedInfo ) {
    std::cout << "Patch order:" << std::endl;
    for ( auto& patch : patches ) {
      std::cout << "Patch[" << patch.getIndex() << "]=(" << patch.getSizeU0() << "," << patch.getSizeV0() << ")"
                << std::endl;
    }
  }

  for ( auto& patch : patches ) { occupancySizeU = ( std::max )( occupancySizeU, patch.getSizeU0() + 1 ); }

  int numROIs     = params_.numROIs_;
  int numTilesHor = params_.numTilesHor_;
  int tileWidth   = occupancySizeU / numTilesHor;
  int tileHeight  = int( tileWidth * params_.tileHeightToWidthRatio_ );
  if ( params_.enablePointCloudPartitioning_ )
    std::cout << "frame " << frame.getFrameIndex() << " tilesize: " << tileWidth << "x" << tileHeight << std::endl;
  occupancySizeV  = ( occupancySizeV >= tileHeight ) ? occupancySizeV : tileHeight;
  int numTilesVer = occupancySizeV / tileHeight;
  // initializating the tile map to -1 (not assigned)
  partitionToTileMap.resize( numTilesHor * numTilesVer, -1 );

  width  = occupancySizeU * params_.occupancyResolution_;
  height = occupancySizeV * params_.occupancyResolution_;
  size_t maxOccupancyRow{0};

  vector<int> orientation_vertical = {
      PATCH_ORIENTATION_DEFAULT, PATCH_ORIENTATION_SWAP,    PATCH_ORIENTATION_ROT180,
      PATCH_ORIENTATION_MIRROR,  PATCH_ORIENTATION_MROT180, PATCH_ORIENTATION_ROT270,
      PATCH_ORIENTATION_MROT90,  PATCH_ORIENTATION_ROT90};  // favoring vertical orientation
  vector<int> orientation_horizontal = {
      PATCH_ORIENTATION_SWAP,   PATCH_ORIENTATION_DEFAULT, PATCH_ORIENTATION_ROT270,
      PATCH_ORIENTATION_MROT90, PATCH_ORIENTATION_ROT90,   PATCH_ORIENTATION_ROT180,
      PATCH_ORIENTATION_MIRROR, PATCH_ORIENTATION_MROT180};  // favoring horizontal orientations (that should be
                                                             // rotated)
  int               numOrientations = params_.useEightOrientations_ ? 8 : 2;
  std::vector<bool> occupancyMap;
  occupancyMap.resize( occupancySizeU * occupancySizeV, false );
  // loop over ROIs
  for ( size_t roiIndex = 0; roiIndex < numROIs; ++roiIndex ) {
    // calculate the position which the tile group will start: top left available tile
    int  tileStartPosU = -1;
    int  tileStartPosV = -1;
    int  tileEndPosU   = -1;
    bool foundLimits   = false;
    while ( !foundLimits ) {
      for ( int v = 0; ( v < ceil( double( occupancySizeV ) / double( tileHeight ) ) ) && !foundLimits; v++ ) {
        for ( int u = 0; ( u < numTilesHor ) && !foundLimits; u++ ) {
          bool tileIsAvailable = ( partitionToTileMap[u + numTilesHor * v] == -1 );
          if ( tileIsAvailable ) {
            tileStartPosU = u;
            tileStartPosV = v;
            // found the top left corner, now keep going to the right to find the end position
            tileEndPosU              = tileStartPosU;
            bool nextTileIsAvailable = true;
            do {
              // check if the next tiles in the horizontal line are available
              if ( tileEndPosU + 1 < numTilesHor ) {
                nextTileIsAvailable = ( partitionToTileMap[( tileEndPosU + 1 ) + numTilesHor * v] == -1 );
                if ( nextTileIsAvailable )
                  tileEndPosU++;
                else
                  nextTileIsAvailable = false;
              } else
                nextTileIsAvailable = false;
            } while ( nextTileIsAvailable );
            foundLimits = true;
          }
        }
      }
      if ( !foundLimits ) {
        // the map might not have any available tiles, let's increase the size of the canvas
        occupancySizeV *= 2;
        occupancyMap.resize( occupancySizeU * occupancySizeV );
        // resizing the partitionToTile as well
        int numTilesVerNew = occupancySizeV / tileHeight;
        partitionToTileMap.resize( numTilesHor * numTilesVerNew );
        for ( int i = numTilesHor * numTilesVer; i < numTilesHor * numTilesVerNew; i++ ) partitionToTileMap[i] = -1;
        numTilesVer = numTilesVerNew;
      }
    }
    // loop over patches of current ROI
    for ( auto& patch : patches ) {
      if ( roiIndex != patch.getRoiIndex() ) { continue; }
      assert( patch.getSizeU0() <= occupancySizeU );
      assert( patch.getSizeV0() <= occupancySizeV );
      bool  locationFound = false;
      auto& occupancy     = patch.getOccupancy();
      while ( !locationFound ) {
        if ( patch.getBestMatchIdx() != InvalidPatchIndex ) {
          patch.getPatchOrientation() = prevPatches[patch.getBestMatchIdx()].getPatchOrientation();
          // try to place on the same position as the matched patch
          patch.getU0() = prevPatches[patch.getBestMatchIdx()].getU0();
          patch.getV0() = prevPatches[patch.getBestMatchIdx()].getV0();
          // try to place the matched patch into the tiles occupied by the
          // current ROI so far
          for ( size_t numTilesInTileVert = 1; numTilesInTileVert <= numTilesVer && !locationFound;
                numTilesInTileVert++ ) {
            for ( size_t numTilesInTileHor = 1;
                  numTilesInTileHor <= ( tileEndPosU - tileStartPosU + 1 ) && !locationFound; numTilesInTileHor++ ) {
              Tile tile;
              tile.minU = (tileStartPosU)*tileWidth;
              tile.maxU = tile.minU + tileWidth * (numTilesInTileHor)-1;
              tile.minV = (tileStartPosV)*tileHeight;
              tile.maxV = tile.minV + tileHeight * (numTilesInTileVert)-1;
              if ( patch.checkFitPatchCanvas( occupancyMap, occupancySizeU, occupancySizeV, params_.lowDelayEncoding_,
                                              safeguard, tile ) ) {
                locationFound = true;
                std::cout << "inter: ROI[" << roiIndex << "] patch " << patch.getIndex() << "\t@(" << patch.getU0()
                          << "," << patch.getV0() << ")\ts(" << patch.getSizeU0() << "x" << patch.getSizeV0() << ")\to"
                          << patch.getPatchOrientation()
                          << " fitted in tile-"
                             "["
                          << tile.minU << "," << tile.maxU << "][" << tile.minV << "," << tile.maxV << "]"
                          << " +-+-+-+-+-(MATCHED patch placed on the same "
                             "position)-+-+-+-+-+"
                          << std::endl;
                if ( printDetailedInfo ) {
                  std::cout << "Maintained orientation " << patch.getPatchOrientation() << " for matched patch "
                            << patch.getIndex() << " in the same position (" << patch.getU0() << "," << patch.getV0()
                            << ")" << std::endl;
                }
              }
            }
          }
          // if the patch couldn't fit, try to fit the patch in the top left position
          if ( !locationFound ) {
            // loop on the size fo the tile group, to fit the patch in the smallest area possible first
            for ( size_t numTilesInTileVert = 1; numTilesInTileVert <= numTilesVer && !locationFound;
                  numTilesInTileVert++ ) {
              for ( size_t numTilesInTileHor = 1;
                    numTilesInTileHor <= ( tileEndPosU - tileStartPosU + 1 ) && !locationFound; numTilesInTileHor++ ) {
                Tile tile;
                tile.minU = (tileStartPosU)*tileWidth;
                tile.maxU = tile.minU + tileWidth * (numTilesInTileHor)-1;
                tile.minV = (tileStartPosV)*tileHeight;
                tile.maxV = tile.minV + tileHeight * (numTilesInTileVert)-1;
                // now look for a possible position in the defined tile group
                for ( size_t v = 0; v < occupancySizeV && !locationFound; ++v ) {
                  for ( size_t u = 0; u < occupancySizeU && !locationFound; ++u ) {
                    patch.getU0() = u;
                    patch.getV0() = v;
                    if ( patch.checkFitPatchCanvas( occupancyMap, occupancySizeU, occupancySizeV,
                                                    params_.lowDelayEncoding_, safeguard, tile ) ) {
                      locationFound = true;
                      std::cout << "inter: ROI[" << roiIndex << "] patch " << patch.getIndex() << "\t@("
                                << patch.getU0() << "," << patch.getV0() << ")\ts(" << patch.getSizeU0() << "x"
                                << patch.getSizeV0() << ")\to" << patch.getPatchOrientation()
                                << " fitted in tile-"
                                   "["
                                << tile.minU << "," << tile.maxU << "][" << tile.minV << "," << tile.maxV << "]"
                                << " +-+-+-+-+-(MATCHED patch placed on new position)-+-+-+-+-+" << std::endl;
                      if ( printDetailedInfo ) {
                        std::cout << "Orientation " << patch.getPatchOrientation() << " selected for patch "
                                  << patch.getIndex() << " (" << u << "," << v << ")" << std::endl;
                      }
                    }
                  }
                }
              }
            }
          }
        } else {
          // best effort
          // loop on the size fo the tile group, to fit the patch in the smallest area possible first
          for ( size_t numTilesInTileVert = 1; numTilesInTileVert <= numTilesVer && !locationFound;
                numTilesInTileVert++ ) {
            for ( size_t numTilesInTileHor = 1;
                  numTilesInTileHor <= ( tileEndPosU - tileStartPosU + 1 ) && !locationFound; numTilesInTileHor++ ) {
              Tile tile;
              tile.minU = (tileStartPosU)*tileWidth;
              tile.maxU = tile.minU + tileWidth * (numTilesInTileHor)-1;
              tile.minV = (tileStartPosV)*tileHeight;
              tile.maxV = tile.minV + tileHeight * (numTilesInTileVert)-1;
              // now look for a possible position in the defined tile group
              for ( size_t v = 0; v < occupancySizeV && !locationFound; ++v ) {
                for ( size_t u = 0; u < occupancySizeU && !locationFound; ++u ) {
                  patch.getU0() = u;
                  patch.getV0() = v;
                  for ( size_t orientationIdx = 0; orientationIdx < numOrientations && !locationFound;
                        orientationIdx++ ) {
                    if ( patch.getSizeU0() > patch.getSizeV0() ) {
                      patch.getPatchOrientation() = orientation_horizontal[orientationIdx];
                    } else {
                      patch.getPatchOrientation() = orientation_vertical[orientationIdx];
                    }
                    if ( patch.checkFitPatchCanvas( occupancyMap, occupancySizeU, occupancySizeV,
                                                    params_.lowDelayEncoding_, safeguard, tile ) ) {
                      locationFound = true;
                      std::cout << "intra: ROI[" << roiIndex << "] patch " << patch.getIndex() << "\t@("
                                << patch.getU0() << "," << patch.getV0() << ")\ts(" << patch.getSizeU0() << "x"
                                << patch.getSizeV0() << ")\to" << patch.getPatchOrientation()
                                << " fitted in tile-"
                                   "["
                                << tile.minU << "," << tile.maxU << "][" << tile.minV << "," << tile.maxV << "]"
                                << std::endl;
                      if ( printDetailedInfo ) {
                        std::cout << "Orientation " << patch.getPatchOrientation() << " selected for patch "
                                  << patch.getIndex() << " (" << u << "," << v << ")" << std::endl;
                      }
                    }
                  }
                }
              }
            }
          }
        }
        if ( !locationFound ) {
          occupancySizeV *= 2;
          occupancyMap.resize( occupancySizeU * occupancySizeV );
          // resizing the partitionToTile as well
          int numTilesVerNew = occupancySizeV / tileHeight;
          partitionToTileMap.resize( numTilesHor * numTilesVerNew );
          for ( int i = numTilesHor * numTilesVer; i < numTilesHor * numTilesVerNew; i++ ) partitionToTileMap[i] = -1;
          numTilesVer = numTilesVerNew;
        }
      }  // while loop
      for ( size_t v0 = 0; v0 < patch.getSizeV0(); ++v0 ) {
        for ( size_t u0 = 0; u0 < patch.getSizeU0(); ++u0 ) {
          int coord = patch.patchBlock2CanvasBlock( u0, v0, occupancySizeU, occupancySizeV );
          if ( params_.lowDelayEncoding_ )
            occupancyMap[coord] = true;
          else
            occupancyMap[coord] = occupancyMap[coord] || occupancy[v0 * patch.getSizeU0() + u0];
          // also claim the tile for the ROI
          size_t x, y;
          patch.patch2Canvas( u0, v0, occupancySizeU * patch.getOccupancyResolution(),
                              occupancySizeV * patch.getOccupancyResolution(), x, y );
          int tileIndex = ( x / patch.getOccupancyResolution() ) / tileWidth +
                          numTilesHor * ( ( y / patch.getOccupancyResolution() ) / tileHeight );
          if ( tileIndex >= partitionToTileMap.size() )
            std::cout << "ERROR" << std::endl;
          else
            partitionToTileMap[tileIndex] = roiIndex;
        }
      }
      if ( !( patch.isPatchDimensionSwitched() ) ) {
        height = ( std::max )( height, ( patch.getV0() + patch.getSizeV0() ) * patch.getOccupancyResolution() );
        width  = ( std::max )( width, ( patch.getU0() + patch.getSizeU0() ) * patch.getOccupancyResolution() );
        maxOccupancyRow = ( std::max )( maxOccupancyRow, ( patch.getV0() + patch.getSizeV0() ) );
      } else {
        height = ( std::max )( height, ( patch.getV0() + patch.getSizeU0() ) * patch.getOccupancyResolution() );
        width  = ( std::max )( width, ( patch.getU0() + patch.getSizeV0() ) * patch.getOccupancyResolution() );
        maxOccupancyRow = ( std::max )( maxOccupancyRow, ( patch.getV0() + patch.getSizeU0() ) );
      }
    }  // patch loop

    if ( printDetailedInfo ) { printMap( occupancyMap, occupancySizeU, occupancySizeV ); }
  }  // ROI loop
  if ( printDetailedInfo ) { printMap( occupancyMap, occupancySizeU, occupancySizeV ); }
  std::cout << "frame " << atlasFrame.getAtlasFrameIndex()
            << " spacialConsistencyPackFlexibleMultipleTiles: actualImageSize " << width << " x " << height
            << std::endl;
}

void PCCEncoder::packTetris( PCCFrameContext& frame, size_t presetWidth, size_t presetHeight, int safeguard ) {
  auto& width   = frame.getWidth();
  auto& height  = frame.getHeight();
  auto& patches = frame.getPatches();
  // set no matched patches, since this function does not take into account the previous frame
  frame.setNumMatchedPatches( 0 );
  if ( patches.empty() ) { return; }
  // sorting by patch largest dimension
  std::sort( patches.begin(), patches.end(), []( PCCPatch& a, PCCPatch& b ) { return a.gt( b ); } );
  if ( printDetailedInfo ) {
    std::cout << "Patch order:" << std::endl;
    for ( auto& patch : patches ) {
      std::cout << "Patch[" << patch.getIndex() << "]=(" << patch.getSizeU0() << "," << patch.getSizeV0() << ")"
                << std::endl;
    }
  }

  size_t occupancySizeU = presetWidth / params_.occupancyResolution_;
  size_t occupancySizeV = ( std::max )( patches[0].getSizeV0(), patches[0].getSizeU0() );
  for ( auto& patch : patches ) { occupancySizeU = ( std::max )( occupancySizeU, patch.getSizeU0() + 1 ); }

  width  = occupancySizeU * params_.occupancyResolution_;
  height = occupancySizeV * params_.occupancyResolution_;
  size_t maxOccupancyRow{0};

  std::vector<bool> occupancyMap;
  occupancyMap.resize( occupancySizeU * occupancySizeV, false );
  std::vector<int> horizon;
  horizon.resize( occupancySizeU, 0 );
  if ( printDetailedInfo ) {
    std::cout << "Horizon :[";
    for ( int i = 0; i < occupancySizeU; i++ ) { std::cout << horizon[i] << ","; }
    std::cout << "]" << std::endl;
  }

  for ( auto& patch : patches ) {
    assert( patch.getSizeU0() <= occupancySizeU );
    assert( patch.getSizeV0() <= occupancySizeV );
    auto& occupancy = patch.getOccupancy();

    // getting the horizons using the rotation 0 position
    if ( printDetailedInfo ) { patch.print(); }
    std::vector<int> top_horizon;
    std::vector<int> bottom_horizon;
    std::vector<int> right_horizon;
    std::vector<int> left_horizon;
    patch.get_patch_horizons( top_horizon, bottom_horizon, right_horizon, left_horizon );

    bool locationFound = false;
    // try to place the patch tetris-style
    vector<int> orientation_values = {
        PATCH_ORIENTATION_DEFAULT, PATCH_ORIENTATION_SWAP,    PATCH_ORIENTATION_ROT180,
        PATCH_ORIENTATION_MIRROR,  PATCH_ORIENTATION_MROT180, PATCH_ORIENTATION_ROT270,
        PATCH_ORIENTATION_MROT90,  PATCH_ORIENTATION_ROT90};  // favoring vertical orientation
    int numOrientations = params_.useEightOrientations_ ? 8 : 2;
    while ( !locationFound ) {
      int    best_wasted_space = ( std::numeric_limits<int>::max )();
      size_t best_u;
      size_t best_v;
      int    best_orientation;
      for ( size_t u = 0; u < occupancySizeU; ++u ) {
        for ( size_t v = 0; v < occupancySizeV; ++v ) {
          patch.getU0() = u;
          patch.getV0() = v;
          for ( size_t orientationIdx = 0; orientationIdx < numOrientations; orientationIdx++ ) {
            patch.getPatchOrientation() = orientation_values[orientationIdx];
            if ( !patch.isPatchLocationAboveHorizon( horizon, top_horizon, bottom_horizon, right_horizon,
                                                     left_horizon ) ) {
              if ( printDetailedInfo ) {
                std::cout << "(" << u << "," << v << "|" << patch.getPatchOrientation() << ") above horizon"
                          << std::endl;
              }
              continue;
            }
            if ( printDetailedInfo ) {
              std::cout << "(" << u << "," << v << "|" << patch.getPatchOrientation() << ")" << std::endl;
            }
            if ( patch.checkFitPatchCanvas( occupancyMap, occupancySizeU, occupancySizeV, params_.lowDelayEncoding_,
                                            safeguard ) ) {
              // now calculate the wasted space
              int wasted_space =
                  patch.calculate_wasted_space( horizon, top_horizon, bottom_horizon, right_horizon, left_horizon );
              if ( printDetailedInfo ) { std::cout << "(wasted space) = " << wasted_space << std::endl; }
              if ( wasted_space < best_wasted_space ) {
                best_wasted_space = wasted_space;
                best_u            = u;
                best_v            = v;
                best_orientation  = patch.getPatchOrientation();
                locationFound     = true;
              }
            }
          }
        }
      }
      if ( !locationFound ) {
        occupancySizeV *= 2;
        occupancyMap.resize( occupancySizeU * occupancySizeV );
        if ( printDetailedInfo ) {
          std::cout << "Increasing frame size (" << occupancySizeU << "," << occupancySizeV << ")" << std::endl;
        }
      } else {
        // select the best position and orientation
        patch.getU0()               = best_u;
        patch.getV0()               = best_v;
        patch.getPatchOrientation() = best_orientation;
        if ( printDetailedInfo ) {
          std::cout << "Selected position (" << best_u << "," << best_v << ") and orientation " << best_orientation
                    << "(wasted space=" << best_wasted_space << ")" << std::endl;
        }
        // update the horizon
        patch.update_horizon( horizon, top_horizon, bottom_horizon, right_horizon, left_horizon );
        // debugging
        if ( printDetailedInfo ) {
          std::cout << "Horizon :[";
          for ( int i = 0; i < occupancySizeU; i++ ) { std::cout << horizon[i] << ","; }
          std::cout << "]" << std::endl;
        }
      }
    }
    for ( size_t v0 = 0; v0 < patch.getSizeV0(); ++v0 ) {
      for ( size_t u0 = 0; u0 < patch.getSizeU0(); ++u0 ) {
        int coord = patch.patchBlock2CanvasBlock( u0, v0, occupancySizeU, occupancySizeV );
        if ( params_.lowDelayEncoding_ ) {
          occupancyMap[coord] = true;
        } else {
          occupancyMap[coord] = occupancyMap[coord] || occupancy[v0 * patch.getSizeU0() + u0];
        }
      }
    }
    if ( !( patch.isPatchDimensionSwitched() ) ) {
      height          = ( std::max )( height, ( patch.getV0() + patch.getSizeV0() ) * patch.getOccupancyResolution() );
      width           = ( std::max )( width, ( patch.getU0() + patch.getSizeU0() ) * patch.getOccupancyResolution() );
      maxOccupancyRow = ( std::max )( maxOccupancyRow, ( patch.getV0() + patch.getSizeV0() ) );
    } else {
      height          = ( std::max )( height, ( patch.getV0() + patch.getSizeU0() ) * patch.getOccupancyResolution() );
      width           = ( std::max )( width, ( patch.getU0() + patch.getSizeV0() ) * patch.getOccupancyResolution() );
      maxOccupancyRow = ( std::max )( maxOccupancyRow, ( patch.getV0() + patch.getSizeU0() ) );
    }

    if ( printDetailedInfo ) { printMapTetris( occupancyMap, occupancySizeU, occupancySizeV, horizon ); }
  }

  if ( frame.getNumberOfRawPointsPatches() > 0 ) {
    packRawPointsPatch( frame, occupancyMap, width, height, occupancySizeU, occupancySizeV, maxOccupancyRow );
  } else {
    if ( printDetailedInfo ) { printMap( occupancyMap, occupancySizeU, occupancySizeV ); }
  }
  if ( params_.enhancedOccupancyMapCode_ && !frame.getUseRawPointsSeparateVideo() ) {
    packEOMTexturePointsPatch( frame, occupancyMap, width, height, occupancySizeU, occupancySizeV, maxOccupancyRow );
  }
  if ( printDetailedInfo ) { printMap( occupancyMap, occupancySizeU, occupancySizeV ); }
  std::cout << "actualImageSize(packTetris) " << width << " x " << height << std::endl;
}

void PCCEncoder::packEOMTexturePointsPatch( PCCFrameContext&   frame,
                                            std::vector<bool>& occupancyMap,
                                            size_t             width,
                                            size_t&            height,
                                            size_t             occupancySizeU,
                                            size_t             occupancySizeV,
                                            size_t             maxOccupancyRow ) {
  if ( !params_.useRawPointsSeparateVideo_ ) { assert( width == frame.getWidth() ); }
  auto&  eomPatches = frame.getEomPatches();
  size_t lastHeight = height;
  for ( size_t i = 0; i < eomPatches.size(); i++ ) {
    auto eomPointsPatchBlocks = static_cast<size_t>(
        ceil( double( eomPatches[i].eomCount_ ) / ( params_.occupancyResolution_ * params_.occupancyResolution_ ) ) );
    auto eomPointsPatchBlocksV = static_cast<size_t>( ceil( double( eomPointsPatchBlocks ) / occupancySizeU ) );
    occupancySizeV += eomPointsPatchBlocksV;
    eomPatches[i].isPatchInAuxVideo_ = params_.useRawPointsSeparateVideo_;
    eomPatches[i].u0_                = 0;
    eomPatches[i].v0_                = lastHeight / params_.occupancyResolution_;
    eomPatches[i].sizeU_             = occupancySizeU;
    eomPatches[i].sizeV_             = eomPointsPatchBlocksV;
    printf(
        "packEOMTexturePointsPatch %zu frame\t%zu tile\t [%zu/%zu] eompatch: pos %zu,%zu(block) size(%zux%zu)(block), "
        "#EOMBlock:%zu #EOM:%zu\n",
        frame.getFrameIndex(), frame.getTileIndex(), i, eomPatches.size(), eomPatches[i].u0_, eomPatches[i].v0_,
        eomPatches[i].sizeU_, eomPatches[i].sizeV_, eomPointsPatchBlocks, eomPatches[i].eomCount_ );
    lastHeight += eomPatches[i].sizeV_ * params_.occupancyResolution_;
  }
  occupancyMap.resize( occupancySizeU * occupancySizeV );
  height = lastHeight;
}

size_t PCCEncoder::packRawPointsPatchSimple( PCCFrameContext& tile,
                                             size_t           patchStartOffsetX,
                                             size_t           patchStartOffsetY ) {
  size_t numberOfRawPointsPatches = tile.getNumberOfRawPointsPatches();
  size_t totalHeight              = 0;
  size_t tileWidth                = tile.getWidth();
  size_t tileWidthInBlock         = tileWidth / params_.occupancyResolution_;
  for ( int i = 0; i < numberOfRawPointsPatches; i++ ) {
    auto& rawPointsPatch       = tile.getRawPointsPatch( i );
    auto  rawPointsPatchBlocks = static_cast<size_t>(
        ceil( double( rawPointsPatch.sizeX() ) / ( params_.occupancyResolution_ * params_.occupancyResolution_ ) ) );
    auto rawPointsPatchBlocksV0 = static_cast<size_t>( ceil( double( rawPointsPatchBlocks ) / tileWidthInBlock ) );
    auto rawPointsPatchBlocksU0 =
        static_cast<size_t>( ceil( double( rawPointsPatchBlocks ) / rawPointsPatchBlocksV0 ) );
    rawPointsPatch.sizeU0_                     = rawPointsPatchBlocksU0;
    rawPointsPatch.sizeV0_                     = rawPointsPatchBlocksV0;
    rawPointsPatch.sizeV_                      = rawPointsPatchBlocksV0 * params_.occupancyResolution_;
    rawPointsPatch.sizeU_                      = rawPointsPatchBlocksU0 * params_.occupancyResolution_;
    std::vector<bool>& rawPointsPatchOccupancy = rawPointsPatch.occupancy_;
    rawPointsPatchOccupancy.resize( rawPointsPatch.sizeU0_ * rawPointsPatch.sizeV0_, false );
    for ( size_t v = 0; v < rawPointsPatch.sizeV_; ++v ) {
      for ( size_t u = 0; u < rawPointsPatch.sizeU_; ++u ) {
        const size_t p = v * rawPointsPatch.sizeU_ + u;
        if ( p < rawPointsPatch.getNumberOfRawPoints() * 3 ) {
          const size_t u0 = u / rawPointsPatch.occupancyResolution_;
          const size_t v0 = v / rawPointsPatch.occupancyResolution_;
          const size_t p0 = v0 * rawPointsPatch.sizeU0_ + u0;
          assert( u0 >= 0 && u0 < rawPointsPatch.sizeU0_ );
          assert( v0 >= 0 && v0 < rawPointsPatch.sizeV0_ );
          rawPointsPatchOccupancy[p0] = true;
        }
      }
    }
    rawPointsPatch.u0_ = 0;
    rawPointsPatch.v0_ = totalHeight / params_.occupancyResolution_;
    totalHeight += rawPointsPatch.sizeV_;

    printf( "packRawPointsPatch[%d/%zu]: posU0V0 %zu,%zu sizeU0V0(%zux%zu) #ofpixels %zu\n", i,
            numberOfRawPointsPatches, rawPointsPatch.u0_, rawPointsPatch.v0_, rawPointsPatch.sizeU0_,
            rawPointsPatch.sizeV0_, rawPointsPatch.sizeX() );
  }
  return totalHeight;
}

size_t PCCEncoder::packRawPointsPatch( PCCFrameContext&   tile,
                                       std::vector<bool>& occupancyMap,
                                       size_t             width,
                                       size_t&            height,
                                       size_t             occupancySizeU,
                                       size_t             occupancySizeV,
                                       size_t             maxOccupancyRow ) {
  size_t numberOfRawPointsPatches = tile.getNumberOfRawPointsPatches();
  size_t safeguard                = 0;

  for ( int i = 0; i < numberOfRawPointsPatches; i++ ) {
    auto& rawPointsPatch       = tile.getRawPointsPatch( i );
    auto  rawPointsPatchBlocks = static_cast<size_t>(
        ceil( double( rawPointsPatch.sizeX() ) / ( params_.occupancyResolution_ * params_.occupancyResolution_ ) ) );
    auto rawPointsPatchBlocksV0 = static_cast<size_t>( ceil( double( rawPointsPatchBlocks ) / occupancySizeU ) );
    auto rawPointsPatchBlocksU0 =
        static_cast<size_t>( ceil( double( rawPointsPatchBlocks ) / rawPointsPatchBlocksV0 ) );
    rawPointsPatch.sizeU0_ = rawPointsPatchBlocksU0;
    rawPointsPatch.sizeV0_ = rawPointsPatchBlocksV0;
    rawPointsPatch.sizeV_  = rawPointsPatchBlocksV0 * params_.occupancyResolution_;
    rawPointsPatch.sizeU_  = rawPointsPatchBlocksU0 * params_.occupancyResolution_;
    PCCPatch patch;
    patch.getSizeU0() = rawPointsPatch.sizeU0_;
    patch.getSizeV0() = rawPointsPatch.sizeV0_;
    patch.getSizeU()  = rawPointsPatch.sizeU_;
    patch.getSizeV()  = rawPointsPatch.sizeV_;
    assert( patch.getSizeU0() <= occupancySizeU );
    std::vector<bool>& patchOccupancy = patch.getOccupancy();
    patchOccupancy.resize( rawPointsPatch.sizeU0_ * rawPointsPatch.sizeV0_, false );

    const int16_t infiniteValue = ( std::numeric_limits<int16_t>::max )();
    rawPointsPatch.resize( rawPointsPatch.sizeU_ * rawPointsPatch.sizeV_, infiniteValue );
    std::vector<bool>& rawPointsPatchOccupancy = rawPointsPatch.occupancy_;
    rawPointsPatchOccupancy.resize( rawPointsPatch.sizeU0_ * rawPointsPatch.sizeV0_, false );

    for ( size_t v = 0; v < rawPointsPatch.sizeV_; ++v ) {
      for ( size_t u = 0; u < rawPointsPatch.sizeU_; ++u ) {
        const size_t p = v * rawPointsPatch.sizeU_ + u;
        if ( p < rawPointsPatch.getNumberOfRawPoints() * 3 ) {
          const size_t u0 = u / rawPointsPatch.occupancyResolution_;
          const size_t v0 = v / rawPointsPatch.occupancyResolution_;
          const size_t p0 = v0 * rawPointsPatch.sizeU0_ + u0;
          assert( u0 >= 0 && u0 < rawPointsPatch.sizeU0_ );
          assert( v0 >= 0 && v0 < rawPointsPatch.sizeV0_ );
          rawPointsPatchOccupancy[p0] = true;
          patchOccupancy[p0]          = true;
        }
      }
    }

    // now placing the raw points patch in the atlas
    bool locationFound = false;
    while ( !locationFound ) {
      patch.getPatchOrientation() = PATCH_ORIENTATION_DEFAULT;
      for ( int v = maxOccupancyRow; v <= occupancySizeV && !locationFound; ++v ) {
        for ( int u = 0; u <= occupancySizeU && !locationFound; ++u ) {
          patch.getU0() = u;
          patch.getV0() = v;
          if ( patch.checkFitPatchCanvas( occupancyMap, occupancySizeU, occupancySizeV, params_.lowDelayEncoding_,
                                          safeguard ) ) {
            locationFound = true;
          }
        }
      }
      if ( !locationFound ) {
        occupancySizeV *= 2;
        occupancyMap.resize( occupancySizeU * occupancySizeV );
      }
    }
    rawPointsPatch.u0_ = patch.getU0();
    rawPointsPatch.v0_ = patch.getV0();
    for ( size_t v0 = 0; v0 < rawPointsPatch.sizeV0_; ++v0 ) {
      const size_t v = rawPointsPatch.v0_ + v0;
      for ( size_t u0 = 0; u0 < rawPointsPatch.sizeU0_; ++u0 ) {
        const size_t u = rawPointsPatch.u0_ + u0;
        if ( params_.lowDelayEncoding_ ) {
          occupancyMap[v * occupancySizeU + u] = true;
        } else {
          occupancyMap[v * occupancySizeU + u] =
              occupancyMap[v * occupancySizeU + u] || rawPointsPatchOccupancy[v0 * rawPointsPatch.sizeU0_ + u0];
        }
      }
      height = ( std::max )( height, ( patch.getV0() + patch.getSizeV0() ) * params_.occupancyResolution_ );
    }
    printf( "packRawPointsPatch[%d/%zu]: posU0V0 %zu,%zu sizeU0V0(%zux%zu) #ofpixels %zu\n", i,
            numberOfRawPointsPatches, rawPointsPatch.u0_, rawPointsPatch.v0_, rawPointsPatch.sizeU0_,
            rawPointsPatch.sizeV0_, rawPointsPatch.sizeX() );
  }
  return height;
}

#define MINIMUM_TH_WEIGHT 0.6

struct mypair {
  int      idx;
  uint32_t value;
};

bool comp1( const mypair& a, const mypair& b ) { return a.value < b.value; }

PCCVector3D PCCEncoder::calculateWeightNormal( size_t geometryBitDepth3D, const PCCPointSet3& source ) {
  size_t            atlasIndex = 0;
  size_t            maxValue   = size_t( 1 ) << geometryBitDepth3D;
  PCCVector3D       weightValue;
  std::vector<bool> pjFace;
  pjFace.resize( maxValue * maxValue * 3 );
  size_t pointCount = source.getPointCount();
  if ( params_.enhancedPP_ ) {
    for ( size_t idx = 0; idx < maxValue * maxValue * 3; idx++ ) { pjFace[idx] = false; }
    const int size_1f = maxValue * maxValue;
    for ( size_t idx = 0; idx < pointCount; idx++ ) {
      const PCCPoint3D point = source[idx];
      int              x;
      int              y;
      // YZ: 0,3
      x                        = int( point[1] );
      y                        = int( point[2] );
      pjFace[y * maxValue + x] = true;

      // ZX: 0,3
      x                                  = int( point[2] );
      y                                  = int( point[0] );
      pjFace[y * maxValue + x + size_1f] = true;

      // XY: 0,3
      x                                      = int( point[0] );
      y                                      = int( point[1] );
      pjFace[y * maxValue + x + size_1f * 2] = true;
    }

    mypair pjCnt[3];
    for ( int x = 0; x < 3; x++ ) {
      pjCnt[x].idx   = x;
      pjCnt[x].value = 0;
    }
    for ( size_t idx = 0; idx < maxValue * maxValue; idx++ ) {
      if ( pjFace[idx] ) { pjCnt[0].value = pjCnt[0].value + 1; }
      if ( pjFace[idx + size_1f] ) { pjCnt[1].value = pjCnt[1].value + 1; }
      if ( pjFace[idx + size_1f * 2] ) { pjCnt[2].value = pjCnt[2].value + 1; }
    }

    std::sort( pjCnt, pjCnt + 3, comp1 );

    double axisWeight[6];
    if ( ( double( pjCnt[0].value ) / double( pjCnt[2].value ) ) >= params_.minWeightEPP_ ) {
      int idx_t         = pjCnt[0].idx;
      axisWeight[idx_t] = axisWeight[idx_t + 3] = double( pjCnt[0].value ) / double( pjCnt[2].value );

      idx_t             = pjCnt[1].idx;
      axisWeight[idx_t] = axisWeight[idx_t + 3] = double( pjCnt[1].value ) / double( pjCnt[2].value );

      idx_t             = pjCnt[2].idx;
      axisWeight[idx_t] = axisWeight[idx_t + 3] = 1.0;
    } else {
      int    idx_t = pjCnt[0].idx;
      double tmpb;
      double tmpa;
      axisWeight[idx_t] = axisWeight[idx_t + 3] = params_.minWeightEPP_;

      idx_t             = pjCnt[2].idx;
      axisWeight[idx_t] = axisWeight[idx_t + 3] = 1.0;

      idx_t = pjCnt[1].idx;
      tmpb  = double( pjCnt[1].value ) / double( pjCnt[2].value );
      tmpa  = double( pjCnt[0].value ) / double( pjCnt[2].value );

      axisWeight[idx_t] = axisWeight[idx_t + 3] =
          params_.minWeightEPP_ + ( tmpb - tmpa ) / ( 1.0 - tmpa ) * ( 1 - params_.minWeightEPP_ );
    }

    weightValue[0] = axisWeight[0];
    weightValue[1] = axisWeight[1];
    weightValue[2] = axisWeight[2];
  } else {
    weightValue[0] = 1.0;
    weightValue[1] = 1.0;
    weightValue[2] = 1.0;
  }
  return weightValue;
}

bool PCCEncoder::generateScaledGeometry( const PCCPointSet3& source, PCCFrameContext& title ) {
  auto& patches = title.getPatches();
  //  const int16_t infiniteDepth = ( std::numeric_limits<int16_t>::max )();
  std::sort( patches.begin(), patches.end() );
  for ( size_t i = 0; i < 3; i++ ) {
    std::vector<int16_t> depth[2];  // depth
    size_t               scaleSizeU = patches[i].getSizeU() / params_.levelOfDetailX_;
    size_t               scaleSizeV = patches[i].getSizeV() / params_.levelOfDetailY_;
    depth[0].resize( scaleSizeU * scaleSizeV );
    depth[1].resize( scaleSizeU * scaleSizeV );

    for ( size_t v = 0; v < scaleSizeV; v++ ) {
      for ( size_t u = 0; u < scaleSizeU; u++ ) {
        size_t p       = v * params_.levelOfDetailY_ * patches[i].getSizeU() + u * params_.levelOfDetailX_;
        size_t pScaled = v * scaleSizeU + u;

        if ( patches[i].getDepth( 0 )[p] == infiniteDepth ) {
          depth[0][pScaled] = depth[1][pScaled] = infiniteDepth;
        } else {
          depth[0][pScaled] = patches[i].getDepth( 0 )[p];
          depth[1][pScaled] = patches[i].getDepth( 1 )[p];
        }
      }
    }

    patches[i].setLodScaleX( params_.levelOfDetailX_ );
    patches[i].setLodScaleYIdc( params_.levelOfDetailY_ );
    patches[i].getSizeU()  = scaleSizeU;
    patches[i].getSizeV()  = scaleSizeV;
    patches[i].getSizeU0() = std::ceil( static_cast<double>( scaleSizeU ) / params_.occupancyResolution_ );
    patches[i].getSizeV0() = std::ceil( static_cast<double>( scaleSizeV ) / params_.occupancyResolution_ );

    patches[i].getOccupancy().clear();
    patches[i].getDepth( 0 ).clear();
    patches[i].getDepth( 1 ).clear();
    patches[i].getOccupancy().resize( patches[i].getSizeU0() * patches[i].getSizeV0(), false );
    patches[i].getDepth( 0 ).resize( scaleSizeU * scaleSizeV );
    patches[i].getDepth( 1 ).resize( scaleSizeU * scaleSizeV );
    for ( size_t v = 0; v < scaleSizeV; v++ ) {
      for ( size_t u = 0; u < scaleSizeU; u++ ) {
        size_t p = v * scaleSizeU + u;
        if ( depth[0][p] == infiniteDepth ) {
          patches[i].getDepth( 0 )[p] = infiniteDepth;
          patches[i].getDepth( 1 )[p] = infiniteDepth;
        } else {
          size_t u0 = u / patches[i].getOccupancyResolution();
          size_t v0 = v / patches[i].getOccupancyResolution();
          size_t p0 = v0 * patches[i].getSizeU0() + u0;
          assert( u0 >= 0 && u0 < patches[i].getSizeU0() );
          assert( v0 >= 0 && v0 < patches[i].getSizeV0() );

          patches[i].getOccupancy()[p0] = true;
          patches[i].getDepth( 0 )[p]   = depth[0][p];
          patches[i].getDepth( 1 )[p]   = depth[1][p];
        }
      }
    }
  }  // i<3

  return true;
}

bool PCCEncoder::generateSegments( const PCCPointSet3&                 source,
                                   PCCAtlasFrameContext&               frameContext,
                                   const PCCPatchSegmenter3Parameters& segmenterParams,
                                   size_t                              frameIndex,
                                   float&                              distanceSrcRec ) {
  if ( source.getPointCount() == 0u ) { return true; }
  auto& frame = frameContext.getTitleFrameContext();
  if ( segmenterParams.additionalProjectionPlaneMode_ != 5 ) {
    auto& patches = frame.getPatches();
    patches.reserve( 256 );
    PCCPatchSegmenter3 segmenter;
    segmenter.setNbThread( params_.nbThread_ );
    segmenter.compute( source, frame.getFrameIndex(), segmenterParams, patches, frame.getSrcPointCloudByPatch(),
                       distanceSrcRec );
  } else if ( segmenterParams.additionalProjectionPlaneMode_ == 5 ) {
    segmentationPartiallyAddtinalProjectionPlane( source, frame, segmenterParams, frameIndex, distanceSrcRec );
  }

  if ( frame.getRawPatchEnabledFlag() ) {
    generateRawPointsPatch( source, frame, segmenterParams.useEnhancedOccupancyMapCode_ );
    for ( int i = 0; i < frame.getNumberOfRawPointsPatches(); i++ ) {
      if ( params_.mortonOrderSortRawPoints_ ) {
        sortRawPointsPatchMorton( frame, i );
      } else {
        sortRawPointsPatch( frame, i );
      }
    }
  }
  if ( params_.enhancedOccupancyMapCode_ ) { generateEomPatch( source, frame ); }
  if ( params_.pointLocalReconstruction_ ) {
    for ( auto& patch : frame.getPatches() ) { patch.getOriginalIndex() = patch.getIndex(); }
  }
  return true;
}

void PCCEncoder::dilateGroupGeometryVideo( PCCContext& context, PCCFrameContext& frame, size_t frameIdx ) {
  auto& videoGeometry         = context.getVideoGeometryMultiple()[0];
  auto& videoGeometryMultiple = context.getVideoGeometryMultiple();
  auto  videoOccupancyMap     = context.getVideoOccupancyMap();
  auto  width                 = frame.getWidth();
  auto  height                = frame.getHeight();
  auto& occupancyMap          = videoOccupancyMap.getFrame( frameIdx );
  auto& frame1 =
      params_.multipleStreams_ ? videoGeometryMultiple[0].getFrame( frameIdx ) : videoGeometry.getFrame( 2 * frameIdx );
  auto& frame2 = params_.multipleStreams_ ? videoGeometryMultiple[1].getFrame( frameIdx )
                                          : videoGeometry.getFrame( 2 * frameIdx + 1 );
  for ( size_t y = 0; y < height; y++ ) {
    for ( size_t x = 0; x < width; x++ ) {
      // const size_t pos = y * width + x;
      if ( occupancyMap.getValue( 0, x / params_.occupancyPrecision_, y / params_.occupancyPrecision_ ) == 0 ) {
        uint32_t avg = ( ( static_cast<uint32_t>( frame1.getValue( 0, x, y ) ) ) +
                         ( static_cast<uint32_t>( frame2.getValue( 0, x, y ) ) ) + 1 ) >>
                       1;
        frame1.setValue( 0, x, y, static_cast<uint16_t>( avg ) );
        frame2.setValue( 0, x, y, static_cast<uint16_t>( avg ) );
      }
    }
  }
}

bool PCCEncoder::generateOccupancyMap( PCCContext& context, bool copyToFrame ) {
  for ( auto& frame : context.getFrames() ) {
    auto& entireFrame = frame.getTitleFrameContext();
    entireFrame.getOccupancyMap().resize( entireFrame.getWidth() * entireFrame.getHeight(), 0 );

    printf( "generateOccupancyMap frame %zu: entireFrameSize:%zux%zu\n", entireFrame.getFrameIndex(),
            entireFrame.getWidth(), entireFrame.getHeight() );
    for ( size_t ti = 0; ti < frame.getNumTilesInAtlasFrame(); ti++ ) {
      auto& tile = frame.getTile( ti );
      generateOccupancyMap( tile );
      if ( params_.enhancedOccupancyMapCode_ ) { modifyOccupancyMapEOM( tile ); }

      if ( copyToFrame ) {
        for ( size_t y = 0; y < tile.getHeight(); y++ ) {
          for ( size_t x = 0; x < tile.getWidth(); x++ ) {
            entireFrame.getOccupancyMap()[( y + tile.getLeftTopYInFrame() ) * entireFrame.getWidth() +
                                          ( x + tile.getLeftTopXInFrame() )] =
                frame.getTile( ti ).getOccupancyMap()[y * tile.getWidth() + x];
          }
        }

      }  // copyToFrame
    }    // ti
    if ( !params_.absoluteD1_ || !params_.absoluteT1_ ) {
      frame.getTitleFrameContext().getFullOccupancyMap() = frame.getTitleFrameContext().getOccupancyMap();
    }
  }
  return true;
}

void PCCEncoder::generateOccupancyMap( PCCFrameContext& tile ) {
  auto& occupancyMap     = tile.getOccupancyMap();
  auto& fullOccupancyMap = tile.getFullOccupancyMap();
  auto& width            = tile.getWidth();
  auto& height           = tile.getHeight();
  occupancyMap.resize( width * height, 0 );
  if ( !params_.absoluteD1_ || !params_.absoluteT1_ ) { fullOccupancyMap.resize( width * height, 0 ); }
  //  const int16_t infiniteDepth = ( std::numeric_limits<int16_t>::max )();
  for ( auto& patch : tile.getPatches() ) {
    for ( size_t v = 0; v < patch.getSizeV(); ++v ) {
      for ( size_t u = 0; u < patch.getSizeU(); ++u ) {
        const size_t  p = v * patch.getSizeU() + u;
        const int16_t d = patch.getDepth( 0 )[p];
        if ( d < infiniteDepth ) {
          size_t x;
          size_t y;
          occupancyMap[patch.patch2Canvas( u, v, width, height, x, y )] = 1;
        }
      }
    }
  }
  if ( !params_.absoluteD1_ || !params_.absoluteT1_ ) { fullOccupancyMap = occupancyMap; }
}

void PCCEncoder::refineOccupancyMap( PCCFrameContext& tile ) {
  auto&        patches          = tile.getPatches();
  const size_t patchCount       = patches.size();
  size_t       countRemove4x4   = 0;
  size_t       countRemove16x16 = 0;

  for ( size_t patchIndex = 0; patchIndex < patchCount; ++patchIndex ) {
    auto& patch = patches[patchIndex];
    // Count number of points in each block 4x4
    if ( params_.occupancyPrecision_ > 1 ) {
      for ( size_t v0 = 0; v0 < patch.getSizeV0(); v0++ ) {
        for ( size_t u0 = 0; u0 < patch.getSizeU0(); u0++ ) {
          for ( size_t v1 = 0; v1 < params_.occupancyResolution_; v1 += params_.occupancyPrecision_ ) {
            for ( size_t u1 = 0; u1 < params_.occupancyResolution_; u1 += params_.occupancyPrecision_ ) {
              size_t countOccupancyMapBlock4x4 = 0;
              for ( size_t v2 = 0; v2 < params_.occupancyPrecision_; v2++ ) {
                const size_t v = v0 * params_.occupancyResolution_ + v1 + v2;
                if ( v < patch.getSizeV() ) {
                  for ( size_t u2 = 0; u2 < params_.occupancyPrecision_; u2++ ) {
                    const size_t u = u0 * params_.occupancyResolution_ + u1 + u2;
                    if ( u < patch.getSizeU() ) {
                      const size_t p = v * patch.getSizeU() + u;
                      if ( patch.getDepth( 0 )[p] < infiniteDepth ) { countOccupancyMapBlock4x4++; }
                    }
                  }
                }
              }
              if ( countOccupancyMapBlock4x4 > 0 ) {
                if ( countOccupancyMapBlock4x4 == 1 ) {
                  countRemove4x4++;
                  for ( size_t v2 = 0; v2 < params_.occupancyPrecision_; v2++ ) {
                    const size_t v = v0 * params_.occupancyResolution_ + v1 + v2;
                    if ( v < patch.getSizeV() ) {
                      for ( size_t u2 = 0; u2 < params_.occupancyPrecision_; u2++ ) {
                        const size_t u = u0 * params_.occupancyResolution_ + u1 + u2;
                        if ( u < patch.getSizeU() ) {
                          const size_t p         = v * patch.getSizeU() + u;
                          patch.getDepth( 0 )[p] = infiniteDepth;
                          patch.getDepth( 1 )[p] = infiniteDepth;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    // Count number of points in each block 16x16
    for ( size_t v0 = 0; v0 < patch.getSizeV0(); v0++ ) {
      for ( size_t u0 = 0; u0 < patch.getSizeU0(); u0++ ) {
        size_t countOccupancyMapBlock16x16 = 0;
        for ( size_t v1 = 0; v1 < params_.occupancyResolution_; ++v1 ) {
          const size_t v = v0 * params_.occupancyResolution_ + v1;
          if ( v < patch.getSizeV() ) {
            for ( size_t u1 = 0; u1 < params_.occupancyResolution_; ++u1 ) {
              const size_t u = u0 * params_.occupancyResolution_ + u1;
              if ( u < patch.getSizeU() ) {
                const size_t p      = v * patch.getSizeU() + u;
                int16_t      depth0 = patch.getDepth( 0 )[p];
                if ( depth0 < infiniteDepth ) { countOccupancyMapBlock16x16++; }
              }
            }
          }
        }
        if ( countOccupancyMapBlock16x16 == 0 ) {
          patch.getOccupancy()[v0 * patch.getSizeU0() + u0] = false;
        } else {
          if ( countOccupancyMapBlock16x16 < 4 && countOccupancyMapBlock16x16 != 0 ) {
            countRemove16x16++;
            patch.getOccupancy()[v0 * patch.getSizeU0() + u0] = false;
            // remove block 16x16
            for ( size_t v1 = 0; v1 < params_.occupancyResolution_; ++v1 ) {
              const size_t v = v0 * params_.occupancyResolution_ + v1;
              if ( v < patch.getSizeV() ) {
                for ( size_t u1 = 0; u1 < params_.occupancyResolution_; ++u1 ) {
                  const size_t u = u0 * params_.occupancyResolution_ + u1;
                  if ( u < patch.getSizeU() ) {
                    const size_t p         = v * patch.getSizeU() + u;
                    patch.getDepth( 0 )[p] = infiniteDepth;
                    patch.getDepth( 1 )[p] = infiniteDepth;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

void PCCEncoder::remove3DMotionEstimationFiles( const std::string& path ) {
  removeFile( path + "occupancy.txt" );
  removeFile( path + "patchInfo.txt" );
  removeFile( path + "blockToPatch.txt" );
}

void PCCEncoder::create3DMotionEstimationFiles( PCCContext& context, const std::string& path ) {
  FILE* occupancyFile    = fopen( ( path + "occupancy.txt" ).c_str(), "wb" );
  FILE* patchInfoFile    = fopen( ( path + "patchInfo.txt" ).c_str(), "wb" );
  FILE* blockToPatchFile = fopen( ( path + "blockToPatch.txt" ).c_str(), "wb" );

  for ( size_t frIdx = 0; frIdx < context.size(); ++frIdx ) {
    auto&        frame              = context.getFrame( frIdx ).getTitleFrameContext();
    auto&        occupancyMapImage  = context.getVideoOccupancyMap().getFrame( frIdx );
    auto&        patches            = frame.getPatches();
    auto&        blockToPatch       = frame.getBlockToPatch();
    const size_t blockToPatchWidth  = frame.getWidth() / params_.occupancyResolution_;
    const size_t blockToPatchHeight = frame.getHeight() / params_.occupancyResolution_;
    fwrite( &blockToPatch[0], sizeof( size_t ), blockToPatchHeight * blockToPatchWidth, blockToPatchFile );
    // fwrite( &occupancyMap[0], sizeof( uint32_t ), frame.getHeight() *
    // frame.getWidth(), occupancyFile );
    uint32_t zeroVal = 0;
    uint32_t oneVal  = 1;
    for ( int y = 0; y < frame.getHeight(); y++ ) {
      for ( int x = 0; x < frame.getWidth(); x++ ) {
        if ( occupancyMapImage.getValue( 0, x / params_.occupancyPrecision_, y / params_.occupancyPrecision_ ) > 0 ) {
          fwrite( &oneVal, sizeof( uint32_t ), 1, occupancyFile );
        } else {
          fwrite( &zeroVal, sizeof( uint32_t ), 1, occupancyFile );
        }
      }
    }
    const size_t numPatches = patches.size();
    fwrite( &numPatches, sizeof( size_t ), 1, patchInfoFile );
    for ( const auto& patch : patches ) {
      size_t projectionIndex = patch.getNormalAxis();
      size_t u0              = patch.getU0();
      size_t v0              = patch.getV0();
      size_t sizeU0          = patch.getSizeU0();
      size_t sizeV0          = patch.getSizeV0();
      size_t d1              = patch.getD1();
      size_t u1              = patch.getU1();
      size_t v1              = patch.getV1();
      fwrite( &projectionIndex, sizeof( size_t ), 1, patchInfoFile );
      fwrite( &u0, sizeof( size_t ), 1, patchInfoFile );
      fwrite( &v0, sizeof( size_t ), 1, patchInfoFile );
      fwrite( &sizeU0, sizeof( size_t ), 1, patchInfoFile );
      fwrite( &sizeV0, sizeof( size_t ), 1, patchInfoFile );
      fwrite( &d1, sizeof( size_t ), 1, patchInfoFile );
      fwrite( &u1, sizeof( size_t ), 1, patchInfoFile );
      fwrite( &v1, sizeof( size_t ), 1, patchInfoFile );
    }
  }
  fclose( blockToPatchFile );
  fclose( occupancyFile );
  fclose( patchInfoFile );
}

void PCCEncoder::generateIntraImage( PCCAtlasFrameContext& atlasFrame,
                                     const size_t          mapIndex,
                                     PCCImageGeometry&     image ) {
  image.resize( atlasFrame.getAtlasFrameWidth(), atlasFrame.getAtlasFrameHeight(), PCCCOLORFORMAT::YUV444 );
  image.set( 0 );
  for ( size_t ti = 0; ti < atlasFrame.getNumTilesInAtlasFrame(); ti++ ) {
    auto&  tile     = atlasFrame.getTile( ti );
    auto   width    = tile.getWidth();
    auto   height   = tile.getHeight();
    size_t maxDepth = 0;
    int    patchIdx = -1;
    for ( auto& patch : tile.getPatches() ) {
      patchIdx++;
      for ( size_t v = 0; v < patch.getSizeV(); ++v ) {
        for ( size_t u = 0; u < patch.getSizeU(); ++u ) {
          const size_t  p = v * patch.getSizeU() + u;
          const int16_t d = patch.getDepth( mapIndex )[p];
          if ( d < infiniteDepth ) {
            size_t x;
            size_t y;
            patch.patch2Canvas( u, v, width, height, x, y );
            assert( x < width && y < height );
            image.setValue( 0, x + tile.getLeftTopXInFrame(), y + tile.getLeftTopYInFrame(), uint16_t( d ) );
            // maxDepth = ( std::max )( maxDepth, patch.getSizeD() );
            maxDepth = ( std::max )( maxDepth, (size_t)d );
          }
        }
      }
    }
    if ( maxDepth >= ( size_t( 1 ) << tile.getGeometry2dBitdepth() ) ) {
      std::cout << "Error: maxDepth(" << maxDepth << ") >=" << ( 1 << tile.getGeometry2dBitdepth() ) << std::endl;
      exit( -1 );
    }

    if ( !tile.getUseRawPointsSeparateVideo() ) {
      size_t numberOfRawPointsPatches = tile.getNumberOfRawPointsPatches();
      for ( int i = 0; i < numberOfRawPointsPatches; i++ ) {
        auto&        rawPointsPatch = tile.getRawPointsPatch( i );
        const size_t v0             = rawPointsPatch.v0_ * rawPointsPatch.occupancyResolution_;
        const size_t u0             = rawPointsPatch.u0_ * rawPointsPatch.occupancyResolution_;
        if ( rawPointsPatch.sizeX() != 0u ) {
          for ( size_t v = 0; v < rawPointsPatch.sizeV_; ++v ) {
            for ( size_t u = 0; u < rawPointsPatch.sizeU_; ++u ) {
              const size_t p = v * rawPointsPatch.sizeU_ + u;
              if ( p < rawPointsPatch.getNumberOfRawPoints() * 3 ) {
                assert( rawPointsPatch.x_[p] < infiniteDepth );
                if ( rawPointsPatch.x_[p] >= infiniteDepth ) {
                  printf( "(rawPointsPatch.x_[%zu] >=infiniteDepth)\n", p );
                  exit( 126 );
                }
                const size_t x = ( u0 + u );
                const size_t y = ( v0 + v );
                assert( x < width && y < height );
                image.setValue( 0, x + tile.getLeftTopXInFrame(), y + tile.getLeftTopYInFrame(),
                                uint16_t( rawPointsPatch.x_[p] ) );
                if ( params_.losslessGeo444_ ) {
                  image.setValue( 1, x + tile.getLeftTopXInFrame(), y + tile.getLeftTopYInFrame(),
                                  uint16_t( rawPointsPatch.y_[p] ) );
                  image.setValue( 2, x + tile.getLeftTopXInFrame(), y + tile.getLeftTopYInFrame(),
                                  uint16_t( rawPointsPatch.z_[p] ) );
                }
              } else {
                const size_t x = ( u0 + u );
                const size_t y = ( v0 + v );
                image.setValue( 0, x + tile.getLeftTopXInFrame(), y + tile.getLeftTopYInFrame(),
                                uint16_t( rawPointsPatch.x_[rawPointsPatch.getNumberOfRawPoints() * 3 - 1] ) );
              }
            }
          }
        }
      }
    }
  }  // tile
}
bool PCCEncoder::predictTextureFrame( PCCFrameContext&       frame,
                                      const PCCImageTexture& reference,
                                      PCCImageTexture&       image ) {
  assert( reference.getWidth() == image.getWidth() );
  assert( reference.getHeight() == image.getHeight() );
  const size_t  refWidth     = reference.getWidth();
  const size_t  refHeight    = reference.getHeight();
  auto&         occupancyMap = frame.getOccupancyMap();
  uint8_t       numBitdepth  = 8;
  const int16_t offsetValue  = ( 1 << ( numBitdepth - 1 ) );
  const int16_t maxValue     = ( 1 << numBitdepth ) - 1;
  for ( size_t y = 0; y < refHeight; ++y ) {
    for ( size_t x = 0; x < refWidth; ++x ) {
      const size_t pos1 = y * refWidth + x;
      if ( occupancyMap[pos1] != 0 ) {
        int16_t reference_color[3];
        if ( reference.getDeprecatedColorFormat() == 0 ) {
          reference_color[0] = reference.getValue( 0, x, y );
          reference_color[1] = reference.getValue( 1, x, y );
          reference_color[2] = reference.getValue( 2, x, y );
        } else {
          /// convert yuv444 (16bit) to normalized yuv444 (format double)
          double y1     = reference.getValue( 0, x, y );
          double u1     = reference.getValue( 1, x, y );
          double v1     = reference.getValue( 2, x, y );
          double offset = 32768.0;
          double scale  = 65535.0;
          double weight = 1.0 / scale;
          y1            = weight * y1;
          u1            = weight * ( u1 - offset );
          v1            = weight * ( v1 - offset );
          y1            = ( std::max )( y1, 0.0 );
          y1            = ( std::min )( y1, 1.0 );
          u1            = ( std::max )( u1, -0.5 );
          u1            = ( std::min )( u1, 0.5 );
          v1            = ( std::max )( v1, -0.5 );
          v1            = ( std::min )( v1, 0.5 );
          //// convert normalized yuv444 to normalized rgb (fromat double)
          double r = y1 /*- 0.00000 * u1*/ + 1.57480 * v1;
          double g = y1 - 0.18733 * u1 - 0.46813 * v1;
          double b = y1 + 1.85563 * u1 /*+ 0.00000 * v1*/;
          //// convert normalized rgb to 8-bit rgb
          reference_color[0] = PCCClip( round( r * 255 ), 0.0, 255.0 );
          reference_color[1] = PCCClip( round( g * 255 ), 0.0, 255.0 );
          reference_color[2] = PCCClip( round( b * 255 ), 0.0, 255.0 );
        }
        for ( size_t c = 0; c < 3; ++c ) {
          const auto    value1 = static_cast<int16_t>( image.getValue( c, x, y ) );
          const int16_t value0 = reference_color[c];
          int16_t       delta  = 0;
          delta                = value1 - value0;
          if ( delta < -offsetValue ) {
            delta = -offsetValue;
          } else if ( delta > offsetValue - 1 ) {
            delta = offsetValue - 1;
          }
          delta += offsetValue;
          delta = delta < 0 ? 0 : ( delta > maxValue ? maxValue : delta );
          image.setValue( c, x, y, static_cast<uint8_t>( delta ) );
        }  // c
      } else {
        image.setValue( 0, x, y, static_cast<uint8_t>( 128 ) );
        image.setValue( 1, x, y, static_cast<uint8_t>( 128 ) );
        image.setValue( 2, x, y, static_cast<uint8_t>( 128 ) );
      }
    }
  }
  return true;
}

bool PCCEncoder::predictGeometryFrame( PCCFrameContext&        frame,
                                       const PCCImageGeometry& reference,
                                       PCCImageGeometry&       image ) {
  assert( reference.getWidth() == image.getWidth() );
  assert( reference.getHeight() == image.getHeight() );

  auto& patches      = frame.getPatches();
  auto& blockToPatch = frame.getBlockToPatch();
  auto& occupancyMap = frame.getFullOccupancyMap();

  const size_t imageWidth  = reference.getWidth();
  const size_t imageHeight = reference.getHeight();

  const size_t blockToPatchWidth  = frame.getWidth() / params_.occupancyResolution_;
  const size_t blockToPatchHeight = frame.getHeight() / params_.occupancyResolution_;

  const size_t patchCount = patches.size();
  size_t       patchIndex{0};
  for ( patchIndex = 0; patchIndex < patchCount; ++patchIndex ) {
    const size_t patchIndexPlusOne = patchIndex + 1;
    auto&        patch             = patches[patchIndex];
    for ( size_t v0 = 0; v0 < patch.getSizeV0(); ++v0 ) {
      for ( size_t u0 = 0; u0 < patch.getSizeU0(); ++u0 ) {
        const size_t blockIndex = patch.patchBlock2CanvasBlock( u0, v0, blockToPatchWidth, blockToPatchHeight );
        if ( blockToPatch[blockIndex] == patchIndexPlusOne ) {
          for ( size_t v1 = 0; v1 < patch.getOccupancyResolution(); ++v1 ) {
            const size_t v = v0 * patch.getOccupancyResolution() + v1;
            for ( size_t u1 = 0; u1 < patch.getOccupancyResolution(); ++u1 ) {
              const size_t u = u0 * patch.getOccupancyResolution() + u1;
              size_t       x;
              size_t       y;
              patch.patch2Canvas( u, v, imageWidth, imageHeight, x, y );
              const bool occupancy = occupancyMap[y * imageWidth + x] != 0;
              if ( !occupancy ) { continue; }
              const auto    value1 = static_cast<uint16_t>( image.getValue( 0, x, y ) );
              const auto    value0 = static_cast<uint16_t>( reference.getValue( 0, x, y ) );
              int_least32_t delta  = 0;
              delta = std::abs( static_cast<int_least32_t>( value1 ) - static_cast<int_least32_t>( value0 ) );
              if ( delta < 0 ) { delta = 0; }
              if ( !params_.losslessGeo_ && delta > 9 ) { delta = 9; }
              image.setValue( 0, x, y, static_cast<uint8_t>( delta ) );
            }
          }
        }
      }
    }
  }
  return true;
}

void PCCEncoder::generateEomPatch( const PCCPointSet3& source, PCCFrameContext& frame ) {
  auto& eomPatches = frame.getEomPatches();
  eomPatches.resize( 1 );
  size_t patchCount    = frame.getPatches().size();
  size_t totalEOMCount = 0;
  std::cout << "eomPointsPatch:==============================" << std::endl;
  for ( size_t patchIdx = 0; patchIdx < patchCount; patchIdx++ ) {
    auto&  patch            = frame.getPatches()[patchIdx];
    size_t eomCountPerPatch = 0;
    totalEOMCount += patch.getEOMCount();
    for ( size_t v = 0; v < patch.getSizeV(); ++v ) {
      for ( size_t u = 0; u < patch.getSizeU(); ++u ) {
        const size_t p       = v * patch.getSizeU() + u;
        int16_t      eomCode = patch.getDepthEnhancedDeltaD()[p];
        if ( eomCode != 0 ) {
          uint16_t nbBits = 0;
          for ( uint16_t i = 0; i < 10; i++ ) {
            if ( ( eomCode & ( 1 << i ) ) != 0 ) { nbBits++; }
          }
          if ( params_.mapCountMinus1_ > 0 ) {
            nbBits--;  // don't count d1
          }
          eomCountPerPatch += nbBits;
        }
      }
    }
    eomPatches[0].memberPatches.push_back( patchIdx );
    eomPatches[0].eomCountPerPatch.push_back( patch.getEOMCount() );
    assert( patch.getEOMCount() == eomCountPerPatch );
    patch.setEOMCount( eomCountPerPatch );
  }
  eomPatches[0].eomCount_ = totalEOMCount;
  std::cout << "\t::numbereOfEomPatch = 1 #point : " << totalEOMCount << std::endl;
#if 0
  for(size_t ii=0; ii<eomPatches[0].memberPatches.size(); ii++){
    std::cout<<"\t\t"<<eomPatches[0].memberPatches[ii]<<"th patch\t"<<
    frame.getPatches()[eomPatches[0].memberPatches[ii]].getTileIndex()<<"\t"<<
    eomPatches[0].eomCountPerPatch[ii]<<std::endl;
  }
#endif
}

void PCCEncoder::generateRawPointsPatch( const PCCPointSet3& source,
                                         PCCFrameContext&    frame,
                                         bool                useEnhancedOccupancyMapCode ) {
  //  const int16_t infiniteDepth    = ( std::numeric_limits<int16_t>::max )();
  auto&        patches = frame.getPatches();
  const size_t geometry3dCoordinatesBitdepth =
      params_.geometry3dCoordinatesBitdepth_ + ( params_.additionalProjectionPlaneMode_ > 0 );

  PCCPointSet3 pointsToBeProjected;
  for ( const auto& patch : patches ) {
    for ( size_t v = 0; v < patch.getSizeV(); ++v ) {
      for ( size_t u = 0; u < patch.getSizeU(); ++u ) {
        const size_t p      = v * patch.getSizeU() + u;
        const size_t depth0 = patch.getDepth( 0 )[p];
        if ( depth0 < infiniteDepth ) {
          PCCPoint3D point0;
          if ( patch.getProjectionMode() == 0 ) {
            point0[patch.getNormalAxis()] = double( depth0 + patch.getD1() );
          } else {
            point0[patch.getNormalAxis()] = double( patch.getD1() - depth0 );
          }
          point0[patch.getTangentAxis()]   = double( u ) + patch.getU1();
          point0[patch.getBitangentAxis()] = double( v ) + patch.getV1();
          if ( patch.getAxisOfAdditionalPlane() != 0 ) {
            PCCPoint3D  input = point0;
            PCCVector3D tmp1;
            PCCPatch::InverseRotatePosition45DegreeOnAxis( patch.getAxisOfAdditionalPlane(),
                                                           geometry3dCoordinatesBitdepth, input, tmp1 );
            point0.x() = tmp1.x();
            point0.y() = tmp1.y();
            point0.z() = tmp1.z();
          }
          pointsToBeProjected.addPoint( point0 );
          if ( useEnhancedOccupancyMapCode ) {
            if ( patch.getDepthEnhancedDeltaD()[p] != 0 ) {
              PCCPoint3D point1;
              point1[patch.getTangentAxis()]   = double( u ) + patch.getU1();
              point1[patch.getBitangentAxis()] = double( v ) + patch.getV1();
              for ( uint16_t i = 0; i < 16; i++ ) {  // surfaceThickness is not necessary here?
                if ( ( patch.getDepthEnhancedDeltaD()[p] & ( 1 << i ) ) != 0 ) {
                  uint16_t nDeltaDCur = ( i + 1 );
                  size_t   depth1     = 0;
                  if ( params_.mapCountMinus1_ == 0 ) {
                    depth1 = depth0;
                    if ( params_.mapCountMinus1_ > 0 ) { depth1 = patch.getDepth( 1 )[p]; }

                    if ( params_.mapCountMinus1_ > 0 && depth0 + nDeltaDCur >= depth1 ) { nDeltaDCur++; }

                  } else {
                    depth1 = patch.getDepth( 1 )[p];
                  }

                  if ( patch.getProjectionMode() == 0 ) {
                    point1[patch.getNormalAxis()] = double( depth0 + patch.getD1() + nDeltaDCur );
                  } else {
                    point1[patch.getNormalAxis()] = double( patch.getD1() - depth0 - nDeltaDCur );
                  }
                  PCCPoint3D input = point1;
                  if ( patch.getAxisOfAdditionalPlane() != 0 ) {
                    PCCVector3D tmp1;
                    PCCPatch::InverseRotatePosition45DegreeOnAxis( patch.getAxisOfAdditionalPlane(),
                                                                   geometry3dCoordinatesBitdepth, input, tmp1 );
                    input.x() = tmp1.x();
                    input.y() = tmp1.y();
                    input.z() = tmp1.z();
                  }
                  pointsToBeProjected.addPoint( input );
                }
              }  // for each i
            }    // if( patch.getDepthEnhancedDeltaD()[p] != 0) )
          } else {
            const size_t depth1 = patch.getDepth( 1 )[p];
            PCCPoint3D   point1;
            point1[patch.getTangentAxis()]   = double( u ) + patch.getU1();
            point1[patch.getBitangentAxis()] = double( v ) + patch.getV1();
            if ( patch.getProjectionMode() == 0 ) {
              point1[patch.getNormalAxis()] = double( depth1 ) + patch.getD1();
            } else {
              point1[patch.getNormalAxis()] = double( patch.getD1() ) - double( depth1 );
            }
            if ( patch.getAxisOfAdditionalPlane() != 0 ) {
              PCCPoint3D  input = point1;
              PCCVector3D tmp3;
              PCCPatch::InverseRotatePosition45DegreeOnAxis( patch.getAxisOfAdditionalPlane(),
                                                             geometry3dCoordinatesBitdepth, input, tmp3 );
              point1.x() = tmp3.x();
              point1.y() = tmp3.y();
              point1.z() = tmp3.z();
            }
            pointsToBeProjected.addPoint( point1 );
          }
        }
      }
    }
  }
  PCCKdTree           kdtreeRawPoints( pointsToBeProjected );
  PCCNNResult         result;
  std::vector<size_t> rawPoints;
  rawPoints.resize( 0 );
  for ( size_t i = 0; i < source.getPointCount(); ++i ) {
    kdtreeRawPoints.search( source[i], 1, result );
    const double dist2 = result.dist( 0 );
    if ( dist2 > 0.0 ) { rawPoints.push_back( i ); }
  }
  size_t numRawPoints = rawPoints.size();

  if ( params_.lossyRawPointsPatch_ ) {
    // Settings for selecting/pruning points.
    const size_t maxNeighborCount                   = 16;
    const size_t maxDist                            = 10;  // lower the value of maxDist, fewer points will selected
    const double minSumOfInvDist4RawPointsSelection = params_.minNormSumOfInvDist4MPSelection_ * maxNeighborCount;
    std::vector<size_t> tmpRawPoints;
    tmpRawPoints.resize( 0 );
    PCCPointSet3 rawPointsSet;
    rawPointsSet.resize( numRawPoints );
    // create raw points cloud
    for ( size_t i = 0; i < numRawPoints; ++i ) { rawPointsSet[i] = source[rawPoints[i]]; }
    PCCKdTree kdtreeRawPointsSet( rawPointsSet );
    double    sumOfInverseDist = 0.0;
    for ( size_t i = 0; i < numRawPoints; ++i ) {
      PCCNNResult result;
      kdtreeRawPointsSet.searchRadius( rawPointsSet[i], maxNeighborCount, maxDist, result );
      sumOfInverseDist = 0.0;
      for ( size_t j = 1; j < result.count(); ++j ) { sumOfInverseDist += 1 / result.dist( j ); }
      if ( sumOfInverseDist >= minSumOfInvDist4RawPointsSelection ) { tmpRawPoints.push_back( rawPoints[i] ); }
    }
    numRawPoints = tmpRawPoints.size();
    rawPoints.resize( numRawPoints );
    rawPoints = tmpRawPoints;
  }

  std::cout << "rawPointsPatch:==============================" << std::endl;
  frame.setTotalNumberOfRawPoints( numRawPoints );
  const int16_t infiniteValue = ( std::numeric_limits<int16_t>::max )();
  frame.setTotalNumberOfRawPoints( rawPoints.size() );
  std::cout << "rawPoints.size() = " << rawPoints.size() << std::endl;
  PCCBox3D inputBbox = source.computeBoundingBox();
  inputBbox.min_.x() = 0;
  inputBbox.min_.y() = 0;
  inputBbox.min_.z() = 0;
  std::cout << "input boundinBox::(min_x, min_y, min_z) = (" << inputBbox.min_.x() << ", " << inputBbox.min_.y() << ", "
            << inputBbox.min_.z() << ");" << std::endl;
  std::cout << "input boundinBox::(max_x, max_y, max_z) = (" << inputBbox.max_.x() << ", " << inputBbox.max_.y() << ", "
            << inputBbox.max_.z() << ");" << std::endl;

  PCCBox3D bboxRawPoints;
  auto     mpsBoxSize = double( 1 << params_.geometryNominal2dBitdepth_ );

  bboxRawPoints.min_ = inputBbox.min_;
  bboxRawPoints.max_ = inputBbox.min_;
  bboxRawPoints.max_.x() += mpsBoxSize;
  bboxRawPoints.max_.y() += mpsBoxSize;
  bboxRawPoints.max_.z() += mpsBoxSize;
  bool   isEmptyBox               = true;
  size_t numberOfRawPointsPatches = 0;

  for ( bboxRawPoints.min_.x() = inputBbox.min_.x(); bboxRawPoints.min_.x() <= inputBbox.max_.x();
        bboxRawPoints.min_.x() += mpsBoxSize ) {
    bboxRawPoints.max_.x() = bboxRawPoints.min_.x() + ( mpsBoxSize - 1 );
    for ( bboxRawPoints.min_.y() = inputBbox.min_.y(); bboxRawPoints.min_.y() <= inputBbox.max_.y();
          bboxRawPoints.min_.y() += mpsBoxSize ) {
      bboxRawPoints.max_.y() = bboxRawPoints.min_.y() + ( mpsBoxSize - 1 );
      for ( bboxRawPoints.min_.z() = inputBbox.min_.z(); bboxRawPoints.min_.z() <= inputBbox.max_.z();
            bboxRawPoints.min_.z() += mpsBoxSize ) {
        bboxRawPoints.max_.z() = bboxRawPoints.min_.z() + ( mpsBoxSize - 1 );
        isEmptyBox             = source.isRawPointsBboxEmpty( rawPoints, bboxRawPoints );
        if ( !isEmptyBox ) {
          std::cout << "Box = ( " << bboxRawPoints.min_.x() << ", " << bboxRawPoints.min_.y() << ", "
                    << bboxRawPoints.min_.z() << ") ~ (" << bboxRawPoints.max_.x() << ", " << bboxRawPoints.max_.y()
                    << ", " << bboxRawPoints.max_.z() << ") " << std::endl;
          numberOfRawPointsPatches++;
          auto&             mpsPatches = frame.getRawPointsPatches();
          PCCRawPointsPatch rawPointsPatch;
          rawPointsPatch.frameIndex_ = frame.getFrameIndex();
          std::vector<size_t> rawPointsBBox;
          source.fillRawPointsBbox( rawPoints, bboxRawPoints, rawPointsBBox );
          const size_t numRawPointsBBox = rawPointsBBox.size();
          frame.getNumberOfRawPoints().resize( numberOfRawPointsPatches );
          frame.setNumberOfRawPoints( numberOfRawPointsPatches - 1, numRawPointsBBox );
          rawPointsPatch.occupancyResolution_ = params_.occupancyResolution_;
          rawPointsPatch.isPatchInAuxVideo_   = params_.useRawPointsSeparateVideo_;
          rawPointsPatch.sizeU_               = 0;
          rawPointsPatch.sizeV_               = 0;
          rawPointsPatch.u0_                  = 0;
          rawPointsPatch.v0_                  = 0;
          rawPointsPatch.sizeV0_              = 0;
          rawPointsPatch.sizeU0_              = 0;
          rawPointsPatch.u1_                  = size_t( bboxRawPoints.min_.x() );
          rawPointsPatch.v1_                  = size_t( bboxRawPoints.min_.y() );
          rawPointsPatch.d1_                  = size_t( bboxRawPoints.min_.z() );
          rawPointsPatch.occupancy_.resize( 0 );
          rawPointsPatch.setNumberOfRawPoints( numRawPointsBBox );
          if ( params_.losslessGeo444_ ) {
            rawPointsPatch.resize( 3 * numRawPointsBBox );
            for ( auto i = 0; i < numRawPointsBBox; ++i ) {
              const PCCPoint3D rawPoints = source[rawPointsBBox[i]];
              rawPointsPatch.x_[i]       = static_cast<uint16_t>( rawPoints.x() - rawPointsPatch.u1_ );
              rawPointsPatch.y_[i]       = static_cast<uint16_t>( rawPoints.y() - rawPointsPatch.v1_ );
              rawPointsPatch.z_[i]       = static_cast<uint16_t>( rawPoints.z() - rawPointsPatch.d1_ );
            }
          } else {
            rawPointsPatch.resize( 3 * numRawPointsBBox );
            for ( auto i = 0; i < numRawPointsBBox; ++i ) {
              const PCCPoint3D rawPoints                  = source[rawPointsBBox[i]];
              rawPointsPatch.x_[i]                        = static_cast<uint16_t>( rawPoints.x() - rawPointsPatch.u1_ );
              rawPointsPatch.x_[numRawPointsBBox + i]     = static_cast<uint16_t>( rawPoints.y() - rawPointsPatch.v1_ );
              rawPointsPatch.x_[2 * numRawPointsBBox + i] = static_cast<uint16_t>( rawPoints.z() - rawPointsPatch.d1_ );
              rawPointsPatch.y_[i]                        = infiniteValue;
              rawPointsPatch.y_[numRawPointsBBox + i]     = infiniteValue;
              rawPointsPatch.y_[2 * numRawPointsBBox + i] = infiniteValue;
              rawPointsPatch.z_[i]                        = infiniteValue;
              rawPointsPatch.z_[numRawPointsBBox + i]     = infiniteValue;
              rawPointsPatch.z_[2 * numRawPointsBBox + i] = infiniteValue;
            }
          }
          mpsPatches.push_back( rawPointsPatch );
          std::cout << "\t::numberOfRawPointsPatches = " << frame.getNumberOfRawPointsPatches()
                    << " #point : " << rawPointsPatch.getNumberOfRawPoints() << " #pixels: " << rawPointsPatch.sizeX()
                    << std::endl;
        }  // not empty
      }
    }
  }
}

void PCCEncoder::sortRawPointsPatchMorton( PCCFrameContext& frame, size_t index ) {
  auto&  rawPointsPatch = frame.getRawPointsPatch( index );
  size_t numRawPoints   = rawPointsPatch.getNumberOfRawPoints();
  if ( numRawPoints != 0u ) {
    PCCPointSet3 rawPointsSet;
    rawPointsSet.resize( numRawPoints );
    for ( size_t i = 0; i < numRawPoints; i++ ) {
      rawPointsSet[i] = params_.losslessGeo444_
                            ? PCCPoint3D( rawPointsPatch.x_[i], rawPointsPatch.y_[i], rawPointsPatch.z_[i] )
                            : PCCPoint3D( rawPointsPatch.x_[i], rawPointsPatch.x_[i + numRawPoints],
                                          rawPointsPatch.x_[i + numRawPoints * 2] );
    }
    // calc Morton code of rawPointsSet
    std::vector<std::pair<uint64_t, PCCPoint3D>> mortonPoint;
    mortonPoint.resize( numRawPoints );
    for ( size_t i = 0; i < numRawPoints; ++i ) {
      mortonPoint[i].first  = mortonAddr( rawPointsSet[i], 0 );
      mortonPoint[i].second = rawPointsSet[i];
    }
    // sort points according to their Morton codes
    std::sort( mortonPoint.begin(), mortonPoint.end() );
    for ( size_t i = 0; i < numRawPoints; ++i ) {
      const PCCPoint3D rawPoints = mortonPoint[i].second;
      if ( params_.losslessGeo444_ ) {
        rawPointsPatch.x_[i] = static_cast<uint16_t>( rawPoints.x() );
        rawPointsPatch.y_[i] = static_cast<uint16_t>( rawPoints.y() );
        rawPointsPatch.z_[i] = static_cast<uint16_t>( rawPoints.z() );
      } else {
        rawPointsPatch.x_[i]                    = static_cast<uint16_t>( rawPoints.x() );
        rawPointsPatch.x_[i + numRawPoints]     = static_cast<uint16_t>( rawPoints.y() );
        rawPointsPatch.x_[i + numRawPoints * 2] = static_cast<uint16_t>( rawPoints.z() );
      }
    }
  }
}

void PCCEncoder::sortRawPointsPatch( PCCFrameContext& frame, size_t index ) {
  auto&        rawPointsPatch       = frame.getRawPointsPatch( index );
  const size_t maxNeighborCount     = 5;
  const size_t neighborSearchRadius = 5 * 5;
  size_t       numRawPoints         = rawPointsPatch.getNumberOfRawPoints();
  if ( numRawPoints != 0u ) {
    vector<size_t> sortIdx;
    sortIdx.reserve( numRawPoints );
    PCCPointSet3 rawPointsSet;
    rawPointsSet.resize( numRawPoints );
    for ( size_t i = 0; i < numRawPoints; i++ ) {
      rawPointsSet[i] = params_.losslessGeo444_
                            ? PCCPoint3D( rawPointsPatch.x_[i], rawPointsPatch.y_[i], rawPointsPatch.z_[i] )
                            : PCCPoint3D( rawPointsPatch.x_[i], rawPointsPatch.x_[i + numRawPoints],
                                          rawPointsPatch.x_[i + numRawPoints * 2] );
    }
    PCCKdTree           kdtreeRawPointsSet( rawPointsSet );
    PCCNNResult         result;
    std::vector<size_t> fifo;
    fifo.reserve( numRawPoints );
    std::vector<bool> flags( numRawPoints, true );

    for ( size_t i = 0; i < numRawPoints; i++ ) {
      if ( flags[i] ) {
        flags[i] = false;
        sortIdx.push_back( i );
        fifo.push_back( i );
        while ( !fifo.empty() ) {
          const size_t currentIdx = fifo.back();
          fifo.pop_back();
          kdtreeRawPointsSet.searchRadius( rawPointsSet[currentIdx], maxNeighborCount, neighborSearchRadius, result );
          for ( size_t j = 0; j < result.count(); j++ ) {
            size_t n = result.indices( j );
            if ( flags[n] ) {
              flags[n] = false;
              sortIdx.push_back( n );
              fifo.push_back( n );
            }
          }
        }
      }
    }

    for ( size_t i = 0; i < numRawPoints; ++i ) {
      const PCCPoint3D rawPoints = rawPointsSet[sortIdx[i]];
      if ( params_.losslessGeo444_ ) {
        rawPointsPatch.x_[i] = static_cast<uint16_t>( rawPoints.x() );
        rawPointsPatch.y_[i] = static_cast<uint16_t>( rawPoints.y() );
        rawPointsPatch.z_[i] = static_cast<uint16_t>( rawPoints.z() );
      } else {
        rawPointsPatch.x_[i]                    = static_cast<uint16_t>( rawPoints.x() );
        rawPointsPatch.x_[i + numRawPoints]     = static_cast<uint16_t>( rawPoints.y() );
        rawPointsPatch.x_[i + numRawPoints * 2] = static_cast<uint16_t>( rawPoints.z() );
      }
    }
  }
}

void PCCEncoder::placeAuxiliaryPointsTiles( PCCContext& context ) {
  std::cout << "placeAuxTileInAuxVideo: width " << context.getAuxVideoWidth() << std::endl;
  size_t maxWidth     = context.getAuxVideoWidth();
  size_t maxHeight    = 0;
  auto&  framesInAFPS = context.getFramesInAFPS();
  size_t numInterval  = 1;
  if ( params_.tileSegmentationType_ == 1 ) numInterval = framesInAFPS.size();
  for ( size_t segIdx = 0; segIdx < numInterval; segIdx++ ) {
    size_t tempMaxTotalHeight = 0;
    size_t firstFrame         = ( params_.tileSegmentationType_ == 1 ) ? framesInAFPS[segIdx].first : 0;
    size_t lastFrame = ( params_.tileSegmentationType_ == 1 ) ? ( framesInAFPS[segIdx].second + 1 ) : context.size();
    size_t numTilesInSeg = ( params_.tileSegmentationType_ == 0 ) ? 1 : context[firstFrame].getNumTilesInAtlasFrame();
#if 1
    std::cout << "sequence group[" << segIdx << "] : " << firstFrame << " ~ " << lastFrame
              << " numTile: " << numTilesInSeg << std::endl;
#endif
    // set height
    size_t maxTileHeight = 0;
    for ( size_t tileIdx = 0; tileIdx < numTilesInSeg; tileIdx++ ) {
      for ( size_t frameIdx = firstFrame; frameIdx < lastFrame; frameIdx++ ) {
        auto&             tile = context[frameIdx].getTile( tileIdx );
        std::vector<bool> auxPointsOccupancyMap;
        size_t            auxPointsOccupancySizeU = maxWidth / params_.occupancyResolution_;
        size_t            auxPointsOccupancySizeV = 1;
        size_t            auxPointsTileHeight     = 0;
        size_t            auxPointsTileWidth      = maxWidth;
        auxPointsOccupancyMap.resize( auxPointsOccupancySizeU * auxPointsOccupancySizeV, false );
        if ( tile.getRawPointsPatches().size() == 0 )
          printf( "packRawPointsPatch[0/0]: none\n" );
        else
          packRawPointsPatch( tile, auxPointsOccupancyMap, auxPointsTileWidth, auxPointsTileHeight,
                              auxPointsOccupancySizeU, auxPointsOccupancySizeV, 0 );
        if ( tile.getEomPatches().size() == 0 )
          printf( "packEOMTexturePointsPatch[0/0]: none\n" );
        else
          packEOMTexturePointsPatch( tile, auxPointsOccupancyMap, auxPointsTileWidth, auxPointsTileHeight,
                                     auxPointsOccupancySizeU, auxPointsOccupancySizeV, 0 );
        auxPointsTileHeight = size_t( std::ceil( double( auxPointsTileHeight ) / 64.0 ) ) * 64;
        context.setAuxTileHeight( tileIdx, std::max( context.getAuxTileHeight( tileIdx ), auxPointsTileHeight ) );
      }
    }
  }
  context.setAuxTileLeftTopY( 0, 0 );
  for ( size_t ti = 1; ti < context.getAuxTileHeight().size(); ti++ ) {
    context.setAuxTileLeftTopY( ti, context.getAuxTileLeftTopY( ti - 1 ) + context.getAuxTileHeight( ti - 1 ) );
  }
#if 1
  std::cout << "placeAuxTileInAuxVideo tile info: ";
  for ( size_t ti = 0; ti < context.getAuxTileHeight().size(); ti++ )
    std::cout << "tile[" << ti << "] LeftTopY: " << context.getAuxTileLeftTopY( ti )
              << "Height: " << context.getAuxTileHeight( ti ) << "\n";

  for ( size_t frameIdx = 0; frameIdx < context.size(); frameIdx++ ) {
    printf( "frame[%zu] video size: %zux%zu\n", frameIdx, context.getAuxVideoWidth(), context.getAuxVideoHeight() );
    for ( size_t tileIdx = 0; tileIdx < context[frameIdx].getNumTilesInAtlasFrame(); tileIdx++ ) {
      auto& rawPatches = context[frameIdx].getTile( tileIdx ).getRawPointsPatches();
      auto& eomPatches = context[frameIdx].getTile( tileIdx ).getEomPatches();
      printf( "tile[%zu] in AuxVideo starts @%zu size %zu\n", tileIdx, context.getAuxTileLeftTopY( tileIdx ),
              context.getAuxTileHeight( tileIdx ) );
      for ( size_t i = 0; i < rawPatches.size(); i++ ) {
        printf( "\trawPatch[%zu] #ofRawPoints %zu, %zu,%zu %zux%zu\n", i, rawPatches[i].getNumberOfRawPoints(),
                rawPatches[i].u0_, rawPatches[i].v0_, rawPatches[i].sizeU0_, rawPatches[i].sizeV0_ );
      }
      for ( size_t i = 0; i < eomPatches.size(); i++ ) {
        printf( "\teomPatch[%zu] #ofEomPoints %zu, %zu,%zu %zux%zu\n", i, eomPatches[i].eomCount_, eomPatches[i].u0_,
                eomPatches[i].v0_, eomPatches[i].sizeU_, eomPatches[i].sizeV_ );
      }
    }
  }
#endif
}
void PCCEncoder::generateRawPointsGeometryVideo( PCCContext& context ) {
  auto& videoRawPointsGeometry = context.getVideoRawPointsGeometry();
  videoRawPointsGeometry.resize( context.size() );
  size_t maxWidth = context.getAuxVideoWidth();
  // context.setAuxVideoWidth( ?? );
  auto&  framesInAFPS = context.getFramesInAFPS();
  size_t numInterval  = 1;
  if ( params_.tileSegmentationType_ == 1 ) numInterval = framesInAFPS.size();

  size_t maxVideoHeight = 0;
  for ( size_t ti = 0; ti < context.getAuxTileHeight().size(); ti++ ) {
    maxVideoHeight += context.getAuxTileHeight( ti );
  }
  for ( size_t fi = 0; fi < context.size(); fi++ ) {
    videoRawPointsGeometry.getFrame( fi ).resize( maxWidth, maxVideoHeight, PCCCOLORFORMAT::YUV444 );
  }

  for ( size_t segIdx = 0; segIdx < numInterval; segIdx++ ) {
    size_t firstFrame = ( params_.tileSegmentationType_ == 1 ) ? framesInAFPS[segIdx].first : 0;
    size_t lastFrame  = ( params_.tileSegmentationType_ == 1 ) ? ( framesInAFPS[segIdx].second + 1 ) : context.size();
    size_t numTilesInSeg = ( params_.tileSegmentationType_ == 0 ) ? 1 : context[firstFrame].getNumTilesInAtlasFrame();
#if 1
    std::cout << "sequence group[" << segIdx << "] : " << firstFrame << " ~ " << lastFrame
              << " numTile: " << numTilesInSeg << std::endl;
#endif
    for ( size_t tileIdx = 0; tileIdx < numTilesInSeg; tileIdx++ ) {
      for ( size_t frameIdx = firstFrame; frameIdx < lastFrame; frameIdx++ ) {
        auto& tile = context[frameIdx].getTile( tileIdx );
        generateRawPointsGeometryImage( context, tile, videoRawPointsGeometry.getFrame( frameIdx ) );
        size_t totalNumRawPoints = 0;
        for ( size_t ii = 0; ii < tile.getNumberOfRawPointsPatches(); ii++ )
          totalNumRawPoints += tile.getRawPointsPatch( ii ).sizeX();
        std::cout << "generate raw Points Video (Geometry) frame[" << frameIdx << "] tile[" << tileIdx << "] : "
                  << "#rawPatches: " << tile.getNumberOfRawPointsPatches()
                  << "#rawPoints(pixels): " << totalNumRawPoints << std::endl;
      }  // frame;
    }    // tiles
  }      // interval
}

void PCCEncoder::generateRawPointsTextureVideo( PCCContext& context ) {
  auto& videoRawPointsTexture = context.getVideoRawPointsTexture();
  videoRawPointsTexture.resize( context.size() );
  for ( size_t frameIdx = 0; frameIdx < context.size(); frameIdx++ ) {
    videoRawPointsTexture.getFrame( frameIdx )
        .resize( context.getAuxVideoWidth(), context.getAuxVideoHeight(), PCCCOLORFORMAT::YUV444 );
  }
  for ( size_t frameIdx = 0; frameIdx < context.size(); frameIdx++ ) {
    for ( size_t tileIdx = 0; tileIdx < context[frameIdx].getNumTilesInAtlasFrame(); tileIdx++ ) {
      auto& tile = context[frameIdx].getTile( tileIdx );
      generateRawPointsTextureImage( context, tile, videoRawPointsTexture.getFrame( frameIdx ) );
      std::cout << "generate raw Points Video (Texture) frame[" << frameIdx << "] tile[" << tileIdx << "] :"
                << " #rawPatches: " << tile.getNumberOfRawPointsPatches()
                << " #rawPoints: " << tile.getTotalNumberOfRawPoints()
                << " #eomPatches: " << tile.getEomPatches().size()
                << " #eomPoints: " << tile.getTotalNumberOfEOMPoints() << std::endl;
    }
  }
  cout << "RawPoints Texture [done]" << endl;
}
void PCCEncoder::generateRawPointsGeometryImage( PCCContext& context, PCCFrameContext& tile, PCCImageGeometry& image ) {
  uint16_t lastValue{0};
  uint16_t lastY{0};
  uint16_t lastZ{0};
  size_t   numberOfRawPointsPatches = tile.getNumberOfRawPointsPatches();
  for ( int i = 0; i < numberOfRawPointsPatches; i++ ) {
    auto&        rawPointsPatch    = tile.getRawPointsPatch( i );
    const size_t v0                = rawPointsPatch.v0_ * rawPointsPatch.occupancyResolution_;
    const size_t u0                = rawPointsPatch.u0_ * rawPointsPatch.occupancyResolution_;
    size_t       numberOfRawPoints = rawPointsPatch.getNumberOfRawPoints();
    printf(
        "\tgenerateRawPointsGeometryImage:: (u0,v0,sizeU,sizeU) = "
        "(%zu,%zu,%zu,%zu) \n",
        u0, v0, rawPointsPatch.sizeU_, rawPointsPatch.sizeV_ );

    rawPointsPatch.isPatchInAuxVideo_ = true;
    if ( params_.losslessGeo444_ ) {
      lastValue = rawPointsPatch.x_[numberOfRawPoints - 1];
      lastY     = rawPointsPatch.y_[numberOfRawPoints - 1];
      lastZ     = rawPointsPatch.z_[numberOfRawPoints - 1];
    } else {
      numberOfRawPoints *= 3;
      lastValue = rawPointsPatch.x_[numberOfRawPoints - 1];
    }
    if ( rawPointsPatch.sizeX() != 0u ) {
      for ( size_t v = 0; v < rawPointsPatch.sizeV_; ++v ) {
        for ( size_t u = 0; u < rawPointsPatch.sizeU_; ++u ) {
          const size_t p = v * rawPointsPatch.sizeU_ + u;
          if ( p < numberOfRawPoints && rawPointsPatch.x_[p] < infiniteDepth ) {
            const size_t x = ( u0 + u );  // always starts at 0
            const size_t y = ( v0 + v ) + context.getAuxTileLeftTopY( tile.getTileIndex() );
            assert( x < context.getAuxVideoWidth() && y < context.getAuxVideoHeight() );
            image.setValue( 0, x, y, uint16_t( rawPointsPatch.x_[p] ) );
            if ( params_.losslessGeo444_ ) {
              image.setValue( 1, x, y, uint16_t( rawPointsPatch.y_[p] ) );
              image.setValue( 2, x, y, uint16_t( rawPointsPatch.z_[p] ) );
            }
          } else {
            const size_t x = ( u0 + u );
            const size_t y = ( v0 + v ) + context.getAuxTileLeftTopY( tile.getTileIndex() );
            image.setValue( 0, x, y, static_cast<uint16_t>( lastValue ) );
            if ( params_.losslessGeo444_ ) {
              image.setValue( 1, x, y, uint16_t( lastY ) );
              image.setValue( 2, x, y, uint16_t( lastZ ) );
            }
          }
        }  // u
      }    // v
    }      // size()!=0
  }
}

void PCCEncoder::generateRawPointsTextureImage( PCCContext& context, PCCFrameContext& tile, PCCImageTexture& image ) {
  bool   losslessAtt              = params_.losslessGeo_;
  size_t numberOfRawPointsPatches = tile.getNumberOfRawPointsPatches();
  size_t numberOfEOMPoints        = tile.getTotalNumberOfEOMPoints();
  size_t numberOfRawPoints        = tile.getTotalNumberOfRawPoints();
  size_t width                    = context.getAuxVideoWidth();

  if ( numberOfRawPoints != 0 ) {
    size_t imageWidthInBlock = width / params_.occupancyResolution_;
    int    rawPatchOffset    = 0;
    for ( int i = 0; i < numberOfRawPointsPatches; i++ ) {
      int    pointIndex           = 0;
      auto&  rawPointsPatch       = tile.getRawPointsPatch( i );
      size_t numRawColorPoints    = rawPointsPatch.getNumberOfRawPoints();
      
      printf(
          "\tgenerateRawPointsTextureImage:: (u0,v0) %zu,%zu, (sizeU,sizeU) %zux%zu\n",
          rawPointsPatch.u0_, rawPointsPatch.v0_, rawPointsPatch.sizeU_, rawPointsPatch.sizeV_ );

      const size_t             v0          = rawPointsPatch.v0_ * rawPointsPatch.occupancyResolution_;
      const size_t             u0          = rawPointsPatch.u0_ * rawPointsPatch.occupancyResolution_;
      std::vector<PCCColor3B>& rawTextures = tile.getRawPointsTextures();

      for ( size_t v = 0; v < rawPointsPatch.sizeV_; ++v ) {
        for ( size_t u = 0; u < rawPointsPatch.sizeU_; ++u ) {
          //          const size_t p = v * rawPointsPatch.sizeU_ + u;
          const size_t x = ( u0 + u );
          const size_t y = ( v0 + v ) + context.getAuxTileLeftTopY( tile.getTileIndex() );
          if ( pointIndex < numRawColorPoints ) {
            image.setValue( 0, x, y, uint16_t( rawTextures[rawPatchOffset + pointIndex].r() ) );
            image.setValue( 1, x, y, uint16_t( rawTextures[rawPatchOffset + pointIndex].g() ) );
            image.setValue( 2, x, y, uint16_t( rawTextures[rawPatchOffset + pointIndex].b() ) );
            pointIndex++;
          } else{
            break;
          }
        }
      }
      rawPatchOffset += numRawColorPoints;
    }  // numberOfRawPointsPatches

    assert( numberOfRawPoints == rawPatchOffset );
  }

  if ( numberOfEOMPoints != 0 ) {
    size_t eomPatchOffset            = 0;
    size_t nPixelInCurrentBlockCount = 0;
    size_t xx = 0, yy = 0;
    double avgR{0.0};
    double avgG{0.0};
    double avgB{0.0};

    for ( auto& eomPointsPatch : tile.getEomPatches() ) {
      std::vector<PCCColor3B>& eomTextures    = tile.getEOMTextures();
      size_t                   patchStartPosX = eomPointsPatch.u0_ * params_.occupancyResolution_;
      size_t                   patchStartPosY = eomPointsPatch.v0_ * params_.occupancyResolution_;

      for ( size_t k = 0; k < eomPointsPatch.eomCount_; k++ ) {
        size_t nBlock = k / 256;
        size_t uBlock = nBlock % ( width / 16 );
        size_t vBlock = nBlock / ( width / 16 );
        xx            = patchStartPosX + uBlock * 16 + ( nPixelInCurrentBlockCount % 16 );
        yy            = patchStartPosY + vBlock * 16 + ( nPixelInCurrentBlockCount / 16 ) +
             context.getAuxTileLeftTopY( tile.getTileIndex() );
        ++nPixelInCurrentBlockCount;
        if ( nPixelInCurrentBlockCount >= 256 ) { nPixelInCurrentBlockCount = 0; }
        image.setValue( 0, xx, yy, eomTextures[k + eomPatchOffset].r() );
        image.setValue( 1, xx, yy, eomTextures[k + eomPatchOffset].g() );
        image.setValue( 2, xx, yy, eomTextures[k + eomPatchOffset].b() );
        avgR = avgR + double( eomTextures[k + eomPatchOffset].r() ) / eomPointsPatch.eomCount_;
        avgG = avgG + double( eomTextures[k + eomPatchOffset].g() ) / eomPointsPatch.eomCount_;
        avgB = avgB + double( eomTextures[k + eomPatchOffset].b() ) / eomPointsPatch.eomCount_;
      }
      eomPatchOffset += eomPointsPatch.eomCount_;
    }  // eomPatches
  }
}

bool PCCEncoder::generateSegments( const PCCGroupOfFrames& sources, PCCContext& context ) {
  PCCPatchSegmenter3Parameters params;
  bool                         res            = true;
  auto&                        frames         = context.getFrames();
  params.nnNormalEstimation_                  = params_.nnNormalEstimation_;
  params.normalOrientation_                   = params_.normalOrientation_;
  params.gridBasedRefineSegmentation_         = params_.gridBasedRefineSegmentation_;
  params.maxNNCountRefineSegmentation_        = params_.maxNNCountRefineSegmentation_;
  params.iterationCountRefineSegmentation_    = params_.iterationCountRefineSegmentation_;
  params.voxelDimensionRefineSegmentation_    = params_.voxelDimensionRefineSegmentation_;
  params.searchRadiusRefineSegmentation_      = params_.searchRadiusRefineSegmentation_;
  params.occupancyResolution_                 = params_.occupancyResolution_;
  params.enablePatchSplitting_                = params_.enablePatchSplitting_;
  params.maxPatchSize_                        = params_.maxPatchSize_;
  params.quantizerSizeX_                      = size_t( 1 ) << params_.log2QuantizerSizeX_;
  params.quantizerSizeY_                      = size_t( 1 ) << params_.log2QuantizerSizeY_;
  params.minPointCountPerCCPatchSegmentation_ = params_.minPointCountPerCCPatchSegmentation_;
  params.maxNNCountPatchSegmentation_         = params_.maxNNCountPatchSegmentation_;
  params.surfaceThickness_                    = params_.surfaceThickness_;
  params.minLevel_                            = params_.minLevel_;
  params.mapCountMinus1_                      = params_.mapCountMinus1_;
  params.maxAllowedDist2RawPointsDetection_   = params_.maxAllowedDist2RawPointsDetection_;
  params.maxAllowedDist2RawPointsSelection_   = params_.maxAllowedDist2RawPointsSelection_;
  params.lambdaRefineSegmentation_            = params_.lambdaRefineSegmentation_;
  params.useEnhancedOccupancyMapCode_         = params_.enhancedOccupancyMapCode_;
  params.absoluteD1_                          = params_.absoluteD1_;
  params.surfaceSeparation_                   = params_.surfaceSeparation_;
  params.additionalProjectionPlaneMode_       = params_.additionalProjectionPlaneMode_;
  params.partialAdditionalProjectionPlane_    = params_.partialAdditionalProjectionPlane_;
  params.maxAllowedDepth_                     = ( size_t( 1 ) << params_.geometryNominal2dBitdepth_ ) - 1;
  params.geometryBitDepth2D_ = params_.geometryNominal2dBitdepth_;
  params.geometryBitDepth3D_ = params_.geometry3dCoordinatesBitdepth_ + ( params_.additionalProjectionPlaneMode_ > 0 );
  params.EOMFixBitCount_     = params_.EOMFixBitCount_;
  params.EOMSingleLayerMode_ = params_.enhancedOccupancyMapCode_ && ( params_.mapCountMinus1_ == 0 );
  params.patchExpansion_     = params_.patchExpansion_;
  params.highGradientSeparation_       = params_.highGradientSeparation_;
  params.minGradient_                  = params_.minGradient_;
  params.minNumHighGradientPoints_     = params_.minNumHighGradientPoints_;
  params.enablePointCloudPartitioning_ = params_.enablePointCloudPartitioning_;
  params.roiBoundingBoxMinX_           = params_.roiBoundingBoxMinX_;
  params.roiBoundingBoxMaxX_           = params_.roiBoundingBoxMaxX_;
  params.roiBoundingBoxMinY_           = params_.roiBoundingBoxMinY_;
  params.roiBoundingBoxMaxY_           = params_.roiBoundingBoxMaxY_;
  params.roiBoundingBoxMinZ_           = params_.roiBoundingBoxMinZ_;
  params.roiBoundingBoxMaxZ_           = params_.roiBoundingBoxMaxZ_;
  params.numTilesHor_                  = params_.numTilesHor_;
  params.tileHeightToWidthRatio_       = params_.tileHeightToWidthRatio_;
  params.numCutsAlong1stLongestAxis_   = params_.numCutsAlong1stLongestAxis_;
  params.numCutsAlong2ndLongestAxis_   = params_.numCutsAlong2ndLongestAxis_;
  params.numCutsAlong3rdLongestAxis_   = params_.numCutsAlong3rdLongestAxis_;
  params.createSubPointCloud_          = params_.pointLocalReconstruction_ || params_.singleMapPixelInterleaving_;
  if ( params_.additionalProjectionPlaneMode_ == 0 || params_.additionalProjectionPlaneMode_ == 5 ) {
    params.weightNormal_ = calculateWeightNormal( params.geometryBitDepth3D_, sources[0] );
  }
  float sumDistanceSrcRec = 0;
  for ( size_t i = 0; i < frames.size(); i++ ) {
    float distanceSrcRec = 0;
    if ( !generateSegments( sources[i], frames[i], params, i, distanceSrcRec ) ) {
      res = false;
      break;
    }
    sumDistanceSrcRec += distanceSrcRec;
  }
  if ( params_.pointLocalReconstruction_ || params_.singleMapPixelInterleaving_ ) {
    const float distanceSrcRec = sumDistanceSrcRec / static_cast<float>( frames.size() );
    if ( distanceSrcRec >= 250.F ) {
      params_.pointLocalReconstruction_   = false;
      params_.mapCountMinus1_             = 1;
      params_.singleMapPixelInterleaving_ = false;
    }
  }
  return res;
}

bool PCCEncoder::placeSegments( const PCCGroupOfFrames& sources, PCCContext& context ) {
  bool res = true;

  if ( params_.tileSegmentationType_ == 1 ) {
    for ( size_t frameIndex = 0; frameIndex < context.getFrames().size(); frameIndex++ ) {
      size_t preIndex = frameIndex > 0 ? ( frameIndex - 1 ) : 0;
      if ( !placeSegments( sources[frameIndex], context[frameIndex], context[preIndex], frameIndex ) ) {
        res = false;
        break;
      }
    }
    resizeGeometryVideo( context, params_.videoEncoderOccupancyCodecId_ );

    generateTilesFromImage( context );

    auto& framesInAFPS = context.getFramesInAFPS();
    segmentSequence( context, framesInAFPS );
    size_t maxNumTile = 0;
    for ( size_t segIdx = 0; segIdx < framesInAFPS.size(); segIdx++ ) {
      size_t firstFrame    = framesInAFPS[segIdx].first;
      size_t lastFrame     = framesInAFPS[segIdx].second;
      size_t numTilesInSeg = context[firstFrame].getNumTilesInAtlasFrame();
      maxNumTile           = std::max( maxNumTile, numTilesInSeg );

      if ( params_.losslessGeo_ && !params_.useRawPointsSeparateVideo_ ) numTilesInSeg -= 1;

      for ( size_t tileIdx = 0; tileIdx < numTilesInSeg; tileIdx++ ) {
        size_t initTileWidth  = context.getFrame( firstFrame ).getTile( tileIdx ).getWidth();
        size_t initTileHeight = context.getFrame( firstFrame ).getTile( tileIdx ).getHeight();

        // resizeTileGeometryVideo( context, tileIdx, initTileWidth, initTileHeight); //do we need this?
        if ( params_.globalPatchAllocation_ > 0 &&
             context.getFrame( firstFrame ).getTile( tileIdx ).getPatches().size() > 0 ) {
          size_t tileWidth  = context.getFrame( firstFrame ).getTile( tileIdx ).getWidth();
          size_t tileHeight = context.getFrame( firstFrame ).getTile( tileIdx ).getHeight();

          std::cout << "GPA starts : tile[" << tileIdx << "]" << std::endl;
          if ( params_.globalPatchAllocation_ == 1 ) {
            performDataAdaptiveGPAMethod( context, tileIdx, tileWidth, tileHeight, firstFrame, lastFrame + 1 );
          } else if ( params_.globalPatchAllocation_ == 2 ) {
            doGlobalTetrisPacking( context, tileIdx, tileWidth, tileHeight, firstFrame, lastFrame + 1 );
          }
          resizeTileGeometryVideo( context, tileIdx, initTileWidth, initTileHeight, firstFrame, lastFrame + 1 );
          std::cout << "\n\t-->after GPA\ttile " << tileIdx << " tileSize resized to"
                    << context.getFrame( firstFrame ).getTile( tileIdx ).getWidth() << " x "
                    << context.getFrame( firstFrame ).getTile( tileIdx ).getHeight() << std::endl;
        }
        if ( params_.maxNumRefAtlasFrame_ != 1 && context.size() > 2 && params_.constrainedPack_ ) {
          adjustReferenceAtlasFrames( context, tileIdx );
        }
      }  // tile
    }    // seg

    if ( params_.enhancedOccupancyMapCode_ && !params_.useRawPointsSeparateVideo_ ) {
      placeEomPatchInTile( context, framesInAFPS );
    }
    for ( size_t segIdx = 0; segIdx < framesInAFPS.size(); segIdx++ ) {
      size_t firstFrame    = framesInAFPS[segIdx].first;
      size_t lastFrame     = framesInAFPS[segIdx].second;
      size_t numTilesInSeg = context[firstFrame].getNumTilesInAtlasFrame();
      if ( ( params_.losslessGeo_ || params_.lossyRawPointsPatch_ ) && !params_.useRawPointsSeparateVideo_ )
        numTilesInSeg -= 1;

      for ( size_t tileIdx = 0; tileIdx < numTilesInSeg; tileIdx++ ) {
        size_t initTileWidth  = context.getFrame( firstFrame ).getTile( tileIdx ).getWidth();
        size_t initTileHeight = context.getFrame( firstFrame ).getTile( tileIdx ).getHeight();
        resizeTileGeometryVideo( context, tileIdx, initTileWidth, initTileHeight, firstFrame, lastFrame + 1 );
      }
    }
    relocateTileGeometryVideo( context, framesInAFPS );
    if ( ( params_.losslessGeo_ || params_.lossyRawPointsPatch_ ) && !params_.useRawPointsSeparateVideo_ ) {
      resizeGeometryVideo( context, params_.videoEncoderOccupancyCodecId_ );
      placeRawPatchTile( context, framesInAFPS );
    }
    resizeGeometryVideo( context, params_.videoEncoderOccupancyCodecId_ );

    for ( size_t frameIdx = 0; frameIdx < context.getFrames().size(); frameIdx++ ) {
      for ( size_t tileIdx = 0; tileIdx < context[frameIdx].getNumTilesInAtlasFrame(); tileIdx++ ) {
        if ( frameIdx == 0 )
          context[frameIdx][tileIdx].setNumRefIdxActive( 0 );
        else
          context[frameIdx][tileIdx].setNumRefIdxActive(
              params_.constrainedPack_ ? std::min( frameIdx, params_.maxNumRefAtlasFrame_ ) : 0 );
      }
    }

    if ( params_.useRawPointsSeparateVideo_ ) {
      context.getAuxTileHeight().resize( maxNumTile );
      context.getAuxTileLeftTopY().resize( maxNumTile );
    }
#if 1
    for ( size_t fi = 0; fi < context.size(); fi++ ) {
      for ( size_t ti = 0; ti < context[fi].getNumTilesInAtlasFrame(); ti++ ) {
        printf( "actual tileSize: frame %zu tile %zu : %zu,%zu (%zux%zu) patchCount %zu\n", fi, ti,
                context[fi].getTile( ti ).getLeftTopXInFrame(), context[fi].getTile( ti ).getLeftTopYInFrame(),
                context[fi].getTile( ti ).getWidth(), context[fi].getTile( ti ).getHeight(),
                context[fi].getTile( ti ).getPatches().size() );
      }
    }
#endif
  } else {
    if ( params_.numMaxTilePerFrame_ > 1 ) generateTilesFromSegments( context );

    for ( size_t tileIdx = 0; tileIdx < params_.numMaxTilePerFrame_; tileIdx++ ) {
      size_t initTileWidth  = context.getFrame( 0 ).getTile( tileIdx ).getWidth();
      size_t initTileHeight = context.getFrame( 0 ).getTile( tileIdx ).getHeight();

      for ( size_t frameIndex = 0; frameIndex < context.size(); frameIndex++ ) {
        if ( sources[frameIndex].getPointCount() == 0u ) { return false; }
        auto&  tile       = context.getFrame( frameIndex ).getTile( tileIdx );
        size_t tileWidth  = tile.getWidth();
        size_t tileHeight = tile.getHeight();
        size_t preIndex   = frameIndex > 0 ? ( frameIndex - 1 ) : 0;
        auto&  prevTile   = context.getFrame( preIndex ).getTile( tileIdx );
        if ( params_.levelOfDetailX_ > 1 || params_.levelOfDetailY_ > 1 ) {
          generateScaledGeometry( sources[frameIndex], tile );
        }

        if ( params_.occupancyMapRefinement_ ) { refineOccupancyMap( tile ); }
        if ( ( frameIndex == 0 ) || ( !params_.constrainedPack_ ) ) {
          if ( params_.packingStrategy_ < 2 ) {
            packFlexible( tile, params_.packingStrategy_, tileWidth, tileHeight, params_.safeGuardDistance_,
                          params_.enablePointCloudPartitioning_ );
          } else if ( params_.packingStrategy_ == 2 ) {
            packTetris( tile, tileWidth, tileHeight, params_.safeGuardDistance_ );
          }
        } else {
          if ( params_.packingStrategy_ < 2 ) {
            if ( params_.globalPatchAllocation_ == 2 ) {
              findMatchesForGlobalTetrisPacking( tile, prevTile );
            } else {
              spatialConsistencyPackFlexible( tile, prevTile, params_.packingStrategy_, tileWidth, tileHeight,
                                              params_.safeGuardDistance_, params_.enablePointCloudPartitioning_ );
            }
          } else if ( params_.packingStrategy_ == 2 ) {
            if ( params_.globalPatchAllocation_ == 2 ) {
              findMatchesForGlobalTetrisPacking( tile, prevTile );  // this could also be a different prevFrame,
              // it depends on the prediction structure
            } else {
              spatialConsistencyPackTetris( tile, prevTile, tileWidth, tileHeight, params_.safeGuardDistance_ );
            }
          }
        }
      }  // frame

      // placing tiles in a frame
      resizeTileGeometryVideo( context, tileIdx, initTileWidth, initTileHeight );
      std::cout << "\t->tile " << tileIdx << " ImageSize " << context.getFrame( 0 ).getTile( tileIdx ).getWidth()
                << " x " << context.getFrame( 0 ).getTile( tileIdx ).getHeight() << std::endl;

      if ( params_.globalPatchAllocation_ > 0 && context.getFrame( 0 ).getTile( tileIdx ).getPatches().size() > 0 ) {
        size_t tileWidth = params_.numMaxTilePerFrame_ == 1 ? params_.minimumImageWidth_
                                                            : context.getFrame( 0 ).getTile( tileIdx ).getWidth();
        size_t tileHeight = params_.numMaxTilePerFrame_ == 1 ? params_.minimumImageHeight_
                                                             : context.getFrame( 0 ).getTile( tileIdx ).getHeight();

        std::cout << "GPA starts : tile[" << tileIdx << "]" << std::endl;
        if ( params_.globalPatchAllocation_ == 1 ) {
          performDataAdaptiveGPAMethod( context, tileIdx, tileWidth, tileHeight );
        } else if ( params_.globalPatchAllocation_ == 2 ) {
          doGlobalTetrisPacking( context, tileIdx, tileWidth, tileHeight );
        }
        resizeTileGeometryVideo( context, tileIdx, initTileWidth, initTileHeight );
        std::cout << "\n\t-->after GPA\ttile " << tileIdx << " ImageSize "
                  << context.getFrame( 0 ).getTile( tileIdx ).getWidth() << " x "
                  << context.getFrame( 0 ).getTile( tileIdx ).getHeight() << std::endl;
      }

      if ( params_.maxNumRefAtlasFrame_ != 1 && context.size() > 2 && params_.constrainedPack_ ) {
        adjustReferenceAtlasFrames( context, tileIdx );
      }

    }  // tile

    resizeGeometryVideo( context, params_.videoEncoderOccupancyCodecId_ );
  }
  return res;
}

bool PCCEncoder::placeSegments( const PCCPointSet3&   source,
                                PCCAtlasFrameContext& atlasFrame,
                                PCCAtlasFrameContext& prevAtlasFrame,
                                size_t                frameIndex ) {
  auto& frame     = atlasFrame.getTitleFrameContext();
  auto& prevFrame = prevAtlasFrame.getTitleFrameContext();

  if ( source.getPointCount() == 0u ) { return false; }
  size_t frameWidth  = params_.minimumImageWidth_;
  size_t frameHeight = params_.minimumImageHeight_;
  if ( params_.levelOfDetailX_ > 1 || params_.levelOfDetailY_ > 1 ) { generateScaledGeometry( source, frame ); }

  if ( params_.occupancyMapRefinement_ ) { refineOccupancyMap( frame ); }

  if ( params_.enablePointCloudPartitioning_ ) {
    if ( params_.packingStrategy_ == 0 ) {
      if ( ( frameIndex == 0 ) || ( !params_.constrainedPack_ ) ) {
        packMultipleTiles( atlasFrame, params_.safeGuardDistance_ );
      } else {
        spatialConsistencyPackMultipleTiles( atlasFrame, prevAtlasFrame, params_.safeGuardDistance_ );
      }
    } else if ( params_.packingStrategy_ == 1 ) {
      if ( ( frameIndex == 0 ) || ( !params_.constrainedPack_ ) ) {
        packFlexibleMultipleTiles( atlasFrame, params_.safeGuardDistance_ );
      } else {
        // assert( params_.globalPatchAllocation_ == 1 );
        spatialConsistencyPackFlexibleMultipleTiles( atlasFrame, prevAtlasFrame, params_.safeGuardDistance_ );
      }
    } else {
      assert( 0 );
    }
    return true;
  }
  if ( params_.packingStrategy_ < 2 ) {
    if ( ( frameIndex == 0 ) || ( !params_.constrainedPack_ ) ) {
      packFlexible( frame, params_.packingStrategy_, frameWidth, frameHeight, params_.safeGuardDistance_,
                    params_.enablePointCloudPartitioning_ );
    } else {
      if ( params_.globalPatchAllocation_ == 2 ) {
        findMatchesForGlobalTetrisPacking( frame, prevFrame );  // this could also be a different prevFrame,
        // it depends on the prediction structure
      } else {
        spatialConsistencyPackFlexible( frame, prevFrame, params_.packingStrategy_, frameWidth, frameHeight,
                                        params_.safeGuardDistance_, params_.enablePointCloudPartitioning_ );
      }
    }
  } else {
    if ( params_.packingStrategy_ == 2 ) {
      if ( ( frameIndex == 0 ) || ( !params_.constrainedPack_ ) ) {
        packTetris( frame, frameWidth, frameHeight, params_.safeGuardDistance_ );
      } else {
        if ( params_.globalPatchAllocation_ == 2 ) {
          findMatchesForGlobalTetrisPacking( frame, prevFrame );  // this could also be a different prevFrame,
          // it depends on the prediction structure
        } else {
          spatialConsistencyPackTetris( frame, prevFrame, frameWidth, frameHeight, params_.safeGuardDistance_ );
        }
      }
    }
  }
  return true;
}
void PCCEncoder::generateTilesFromSegments( PCCContext& context ) {
  size_t minimumTileWidth = std::ceil( ( params_.minimumImageWidth_ * 0.1 ) / 64.0 ) * 64;
  size_t minSize0 = 0, maxSize0 = 0;
  for ( size_t fi = 0; fi < context.size(); fi++ ) {
    auto& patchSegmentationFrame = context[fi].getTitleFrameContext();
    auto& patches                = patchSegmentationFrame.getPatches();
    std::sort( patches.begin(), patches.end(), []( PCCPatch& a, PCCPatch& b ) { return a.gt( b ); } );

    if ( fi == 0 )
      minSize0 = std::min( patches[patches.size() - 1].getSizeU0(), patches[patches.size() - 1].getSizeV0() );
    for ( auto& patch : patches ) {
      maxSize0 = std::max( maxSize0, std::max( patch.getSizeU0(), patch.getSizeV0() ) );
      minSize0 = std::min( minSize0, std::min( patch.getSizeU0(), patch.getSizeV0() ) );
    }
  }

  minimumTileWidth  = std::max( minimumTileWidth, minSize0 * params_.occupancyResolution_ );
  size_t tile0Width = std::ceil( static_cast<double>( params_.minimumImageWidth_ - minimumTileWidth ) / 64.0 ) * 64;
  minimumTileWidth  = params_.minimumImageWidth_ - tile0Width;
  std::vector<size_t> tileHeight;

  for ( size_t fi = 0; fi < context.size(); fi++ ) {
    tileHeight.resize( context[fi].getNumTilesInAtlasFrame() );
    auto& patchSegmentationFrame = context[fi].getTitleFrameContext();
    auto& patches                = patchSegmentationFrame.getPatches();
    std::sort( patches.begin(), patches.end(), []( PCCPatch& a, PCCPatch& b ) { return a.gt( b ); } );

    context[fi].getTile( 0 ).setWidth( tile0Width );
    context[fi].getTile( 1 ).setWidth( minimumTileWidth );
    context[fi].getTile( 2 ).setWidth( tile0Width );

    context[fi].getTile( 0 ).setTileIndex( 0 );
    context[fi].getTile( 1 ).setTileIndex( 1 );
    context[fi].getTile( 2 ).setTileIndex( 2 );

    PCCEomPatch eomPatch[3];
    eomPatch[0].eomCount_ = eomPatch[1].eomCount_ = eomPatch[2].eomCount_ = 0;

    size_t numPatchInTile0 = std::min( size_t( 3 ), patches.size() );
    tileHeight[0]          = 0;
    for ( size_t patchIdx = 0; patchIdx < numPatchInTile0; patchIdx++ ) {
      patches[patchIdx].setTileIndex( 0 );
      patches[patchIdx].setFrameIndex( fi );
      context[fi].getTile( 0 ).getPatches().push_back( patches[patchIdx] );
      tileHeight[0] =
          std::max( tileHeight[0], std::max( patches[patchIdx].getSizeU0() * params_.occupancyResolution_,
                                             patches[patchIdx].getSizeV0() * params_.occupancyResolution_ ) );
      if ( params_.enhancedOccupancyMapCode_ ) {
        eomPatch[0].memberPatches.push_back( patchIdx );
        eomPatch[0].eomCountPerPatch.push_back( patches[patchIdx].getEOMCount() );
        eomPatch[0].eomCount_ += patches[patchIdx].getEOMCount();
      }  // if eom enabled
    }    // group0
    for ( size_t patchIdx = numPatchInTile0; patchIdx < patches.size(); patchIdx++ ) {
      patches[patchIdx].setFrameIndex( fi );
      if ( std::max( patches[patchIdx].getSizeU0() * patches[patchIdx].getOccupancyResolution(),
                     patches[patchIdx].getSizeV0() * patches[patchIdx].getOccupancyResolution() ) < minimumTileWidth ) {
        patches[patchIdx].setTileIndex( 1 );
        context[fi].getTile( 1 ).getPatches().push_back( patches[patchIdx] );
        tileHeight[1] =
            std::max( tileHeight[1], std::max( patches[patchIdx].getSizeU0() * params_.occupancyResolution_,
                                               patches[patchIdx].getSizeV0() * params_.occupancyResolution_ ) );

        if ( params_.enhancedOccupancyMapCode_ ) {
          eomPatch[1].memberPatches.push_back( context[fi].getTile( 1 ).getPatches().size() -
                                               1 );  // patchorder in Group1
          eomPatch[1].eomCountPerPatch.push_back( patches[patchIdx].getEOMCount() );
          eomPatch[1].eomCount_ += patches[patchIdx].getEOMCount();
        }  // if eom enabled
      } else {
        patches[patchIdx].setTileIndex( 2 );
        context[fi].getTile( 2 ).getPatches().push_back( patches[patchIdx] );
        tileHeight[2] =
            std::max( tileHeight[2], std::max( patches[patchIdx].getSizeU0() * params_.occupancyResolution_,
                                               patches[patchIdx].getSizeV0() * params_.occupancyResolution_ ) );

        if ( params_.enhancedOccupancyMapCode_ ) {
          eomPatch[2].memberPatches.push_back( context[fi].getTile( 2 ).getPatches().size() -
                                               1 );  // patchorder in Group2
          eomPatch[2].eomCountPerPatch.push_back( patches[patchIdx].getEOMCount() );
          eomPatch[2].eomCount_ += patches[patchIdx].getEOMCount();
        }  // if eom enabled
      }
    }  // patches
    // no non-zero tile
    if ( context[fi].getTile( 1 ).getPatches().size() == 0 || context[fi].getTile( 2 ).getPatches().size() == 0 ) {
      auto& desTile =
          ( context[fi].getTile( 1 ).getPatches().size() == 0 ) ? context[fi].getTile( 1 ) : context[fi].getTile( 2 );
      auto& srcTile =
          ( context[fi].getTile( 1 ).getPatches().size() == 0 ) ? context[fi].getTile( 2 ) : context[fi].getTile( 1 );
      desTile.getPatches().push_back( srcTile.getPatches()[srcTile.getPatches().size() - 1] );
      srcTile.getPatches().pop_back();
    }

    context[fi].getTile( 0 ).setHeight( tileHeight[0] );
    context[fi].getTile( 1 ).setHeight( tileHeight[1] );
    context[fi].getTile( 2 ).setHeight( tileHeight[2] );

    if ( params_.enhancedOccupancyMapCode_ ) {
      context[fi].getTile( 0 ).getEomPatches().push_back( eomPatch[0] );
      context[fi].getTile( 1 ).getEomPatches().push_back( eomPatch[1] );
      context[fi].getTile( 2 ).getEomPatches().push_back( eomPatch[2] );
    }

    if ( ( params_.losslessGeo_ || params_.lossyRawPointsPatch_ ) ) {
      context[fi].getTile( 0 ).getRawPointsPatches().clear();
      context[fi].getTile( 1 ).getRawPointsPatches().clear();
      context[fi].getTile( 2 ).getRawPointsPatches().clear();

      // rawpatches are in a seperate tile
      context[fi].getTile( 3 ).setWidth( params_.minimumImageWidth_ );
      context[fi].getTile( 3 ).setTileIndex( 3 );
      context[fi].getTile( 3 ).getPatches().clear();
      auto& rawPatches = patchSegmentationFrame.getRawPointsPatches();
      for ( size_t patchIdx = 0; patchIdx < rawPatches.size(); patchIdx++ ) {
        rawPatches[patchIdx].tileIndex_  = 3;
        rawPatches[patchIdx].frameIndex_ = fi;
        context[fi].getTile( 3 ).getRawPointsPatches().push_back( rawPatches[patchIdx] );
      }  // rawPatch
      context[fi].getTile( 3 ).setTotalNumberOfRawPoints( patchSegmentationFrame.getTotalNumberOfRawPoints() );
    }

#if 1
    printf( "generateTilesFromSegments: tile[0] : %zux%zu, %zu patches\n", context[fi].getTile( 0 ).getWidth(),
            context[fi].getTile( 0 ).getHeight(), context[fi].getTile( 0 ).getPatches().size() );
    printf( "generateTilesFromSegments: tile[1] : %zux%zu, %zu patches\n", context[fi].getTile( 1 ).getWidth(),
            context[fi].getTile( 1 ).getHeight(), context[fi].getTile( 1 ).getPatches().size() );
    printf( "generateTilesFromSegments: tile[2] : %zux%zu, %zu patches\n", context[fi].getTile( 2 ).getWidth(),
            context[fi].getTile( 2 ).getHeight(), context[fi].getTile( 2 ).getPatches().size() );
    if ( ( params_.losslessGeo_ || params_.lossyRawPointsPatch_ ) ) {
      printf( "generateTilesFromSegments: tile[3] : %zux%zu, %zu patches, %zu rawPatches\n",
              context[fi].getTile( 3 ).getWidth(), context[fi].getTile( 3 ).getHeight(),
              context[fi].getTile( 3 ).getPatches().size(), context[fi].getTile( 3 ).getRawPointsPatches().size() );
    }
    if ( context[fi].getTile( 0 ).getWidth() == 0 || context[fi].getTile( 0 ).getWidth() == 0 ||
         context[fi].getTile( 0 ).getWidth() == 0 || context[fi].getTile( 0 ).getHeight() == 0 ||
         context[fi].getTile( 0 ).getHeight() == 0 || context[fi].getTile( 0 ).getHeight() == 0 )
      exit( 254 );
#endif

  }  // fi
}

void PCCEncoder::generateTilesFromImage( PCCContext& context ) {
  for ( size_t frameIndex = 0; frameIndex < context.getFrames().size(); frameIndex++ ) {
    auto&  frameContainer     = context[frameIndex];
    auto&  frame              = frameContainer.getTitleFrameContext();
    auto&  partitionToTileMap = frameContainer.getPartitionToTileMap();
    int    maxTileID          = *( std::max_element( partitionToTileMap.begin(), partitionToTileMap.end() ) );
    size_t numTiles           = ( std::max )( 0, maxTileID ) + 1;

    frameContainer.setAtlasFrameWidth( frame.getWidth() );
    frameContainer.setAtlasFrameHeight( frame.getHeight() );
    size_t partitionWidthIn64 = ( frameContainer.getAtlasFrameWidth() / ( 64 * params_.numTilesHor_ ) );
    size_t partitionHeightIn64 =
        ( params_.tileHeightToWidthRatio_ * frameContainer.getAtlasFrameWidth() / ( 64 * params_.numTilesHor_ ) );

    bool bSinglePartitionPerTile = ( numTiles == partitionToTileMap.size() );

    frameContainer.initNumTiles( numTiles );
    frameContainer.updatePartitionInfoPerFrame(
        frameIndex, frameContainer.getTitleFrameContext().getWidth(), frameContainer.getTitleFrameContext().getHeight(),
        numTiles, true, partitionWidthIn64, partitionHeightIn64, -1, -1, bSinglePartitionPerTile, false );

    for ( uint32_t tileIdx = 0; tileIdx < numTiles; tileIdx++ ) {
      auto& tile = frameContainer.getTile( tileIdx );
      tile.setFrameIndex( frameIndex );
      size_t tileLeftXinBlock = frameContainer.getAtlasFrameWidth();
      size_t tileLeftYinBlock = frameContainer.getAtlasFrameHeight();
      size_t tileWidth        = 0;
      size_t tileHeight       = 0;
      for ( size_t pi = 0; pi < frameContainer.getTitleFrameContext().getPatches().size(); pi++ ) {
        auto patch = frameContainer.getTitleFrameContext().getPatch( pi );
        if ( patch.getRoiIndex() == tileIdx ) {
          tile.getPatches().push_back( patch );
          auto& curTilePatch = tile.getPatch( tile.getPatches().size() - 1 );
          curTilePatch.setIndexInFrame( pi );
          curTilePatch.setTileIndex( tileIdx );

          tileLeftXinBlock = std::min( tileLeftXinBlock, curTilePatch.getU0() );
          tileLeftYinBlock = std::min( tileLeftYinBlock, curTilePatch.getV0() );

          bool nonRotation = curTilePatch.getPatchOrientation() == PATCH_ORIENTATION_DEFAULT ||
                             curTilePatch.getPatchOrientation() == PATCH_ORIENTATION_ROT180 ||
                             curTilePatch.getPatchOrientation() == PATCH_ORIENTATION_MIRROR ||
                             curTilePatch.getPatchOrientation() == PATCH_ORIENTATION_MROT180;

          size_t patchImageWidthInBlock  = nonRotation ? curTilePatch.getSizeU0() : curTilePatch.getSizeV0();
          size_t patchImageHeightInBlock = nonRotation ? curTilePatch.getSizeV0() : curTilePatch.getSizeU0();
          tileWidth                      = std::max(
              tileWidth, ( curTilePatch.getU0() + patchImageWidthInBlock ) * curTilePatch.getOccupancyResolution() );
          tileHeight = std::max(
              tileHeight, ( curTilePatch.getV0() + patchImageHeightInBlock ) * curTilePatch.getOccupancyResolution() );
        }
      }  // pi

      for ( auto& curTilePatch : tile.getPatches() ) {
        curTilePatch.getU0() -= tileLeftXinBlock;
        curTilePatch.getV0() -= tileLeftYinBlock;
      }
      tileWidth -= tileLeftXinBlock * params_.occupancyResolution_;
      tileHeight -= tileLeftYinBlock * params_.occupancyResolution_;
      tile.setWidth( tileWidth );
      tile.setHeight( tileHeight );
      tile.setLeftTopXInFrame( tileLeftXinBlock * params_.occupancyResolution_ );
      tile.setLeftTopYInFrame( tileLeftYinBlock * params_.occupancyResolution_ );
      tile.getOccupancyMap().resize( tileWidth * tileHeight );
      tile.getFullOccupancyMap().resize( tileWidth * tileHeight );

      if ( frameIndex != 0 && params_.constrainedPack_ ) {
        size_t prevFrameIndex = frameIndex - 1;
        auto&  prevTile       = context[prevFrameIndex].getTile( tileIdx );
        for ( size_t pi = 0; pi < tile.getPatches().size(); pi++ ) {
          auto& patch = tile.getPatch( pi );
          if ( patch.getBestMatchIdx() != InvalidPatchIndex ) {
            for ( size_t piPrev = 0; piPrev < prevTile.getPatches().size(); piPrev++ ) {
              if ( prevTile.getPatch( piPrev ).getIndexInFrame() == patch.getBestMatchIdx() ) {
                patch.setBestMatchIdx( piPrev );
              }  // if
            }    // piPreve
          }      // not-invalid
        }        // pi
      }          // if(params_contrainedPack_)

      if ( params_.enhancedOccupancyMapCode_ ) {
        PCCEomPatch eomPatch;
        eomPatch.tileIndex_  = tile.getTileIndex();
        eomPatch.frameIndex_ = tile.getFrameIndex();
        eomPatch.eomCount_   = 0;
        auto& patches        = tile.getPatches();
        for ( size_t patchIdx = 0; patchIdx < patches.size(); patchIdx++ ) {
          if ( patches[patchIdx].getEOMCount() != 0 ) {
            eomPatch.memberPatches.push_back( patchIdx );
            eomPatch.eomCountPerPatch.push_back( patches[patchIdx].getEOMCount() );
            eomPatch.eomCount_ += patches[patchIdx].getEOMCount();
          }
        }
        tile.getEomPatches().push_back( eomPatch );
      }

      tile.setRawPatchEnabledFlag( params_.losslessGeo_ || params_.lossyRawPointsPatch_ );
      tile.setUseRawPointsSeparateVideo( params_.useRawPointsSeparateVideo_ );
      tile.setGeometry3dCoordinatesBitdepth( params_.geometry3dCoordinatesBitdepth_ +
                                             ( params_.additionalProjectionPlaneMode_ > 0 ) );
      tile.setGeometry2dBitdepth( params_.geometryNominal2dBitdepth_ );
      tile.setMaxDepth( ( 1 << params_.geometryNominal2dBitdepth_ ) - 1 );
      tile.setLog2PatchQuantizerSizeX( context.getLog2PatchQuantizerSizeX() );
      tile.setLog2PatchQuantizerSizeY( context.getLog2PatchQuantizerSizeY() );
      tile.setAtlasFrmOrderCntLsb( context.calculateAFOCLsb( frameIndex ) );
      tile.setAtlasFrmOrderCntVal( frameIndex );
      tile.setFrameIndex( frameIndex );
      tile.setTileIndex( tileIdx );
      tile.setRefAfocList( context, tile.getBestRefListIndexInAsps() );
    }  // tile
    if ( params_.losslessGeo_ || params_.lossyRawPointsPatch_ ) {
      context[frameIndex].setNumTilesInAtlasFrame( context[frameIndex].getNumTilesInAtlasFrame() + 1 );
      context[frameIndex].getTiles().resize( context[frameIndex].getNumTilesInAtlasFrame() );
      auto& tile = context[frameIndex].getTile( context[frameIndex].getNumTilesInAtlasFrame() - 1 );
      tile.setTileIndex( context[frameIndex].getNumTilesInAtlasFrame() - 1 );
      tile.setRawPatchEnabledFlag( params_.losslessGeo_ || params_.lossyRawPointsPatch_ );
      tile.setUseRawPointsSeparateVideo( params_.useRawPointsSeparateVideo_ );
      tile.setGeometry3dCoordinatesBitdepth( params_.geometry3dCoordinatesBitdepth_ +
                                             ( params_.additionalProjectionPlaneMode_ > 0 ) );
      tile.setGeometry2dBitdepth( params_.geometryNominal2dBitdepth_ );
      tile.setMaxDepth( ( 1 << params_.geometryNominal2dBitdepth_ ) - 1 );
      tile.setLog2PatchQuantizerSizeX( context.getLog2PatchQuantizerSizeX() );
      tile.setLog2PatchQuantizerSizeY( context.getLog2PatchQuantizerSizeY() );
      tile.setAtlasFrmOrderCntLsb( context.calculateAFOCLsb( frameIndex ) );
      tile.setAtlasFrmOrderCntVal( frameIndex );
      tile.setFrameIndex( frameIndex );
      tile.setRefAfocList( context, tile.getBestRefListIndexInAsps() );
      tile.setNumberOfRawPointsPatches( context[frameIndex].getTitleFrameContext().getNumberOfRawPointsPatches() );
      tile.setTotalNumberOfRawPoints( context[frameIndex].getTitleFrameContext().getTotalNumberOfRawPoints() );
      for ( size_t rawPatchIdx = 0;
            rawPatchIdx < context[frameIndex].getTitleFrameContext().getNumberOfRawPointsPatches(); rawPatchIdx++ ) {
        auto rawPointsPatch       = context[frameIndex].getTitleFrameContext().getRawPointsPatch( rawPatchIdx );
        rawPointsPatch.tileIndex_ = tile.getTileIndex();
        tile.getRawPointsPatches().push_back( rawPointsPatch );
      }  // rawpatches
      context[frameIndex].getTitleFrameContext().getRawPointsPatches().clear();
    }
  }  // frameIndex
}

void PCCEncoder::placeTiles( PCCContext& context, size_t minFrameWidth, size_t minFrameHeight ) {
  size_t frameWidth  = 0;
  size_t frameHeight = 0;

  if ( params_.tileSegmentationType_ == 1 ) {
  } else {
    // packing
    for ( size_t frameIdx = 0; frameIdx < context.size(); frameIdx++ ) {
      // tile0 : (start from 0,0)
      context[frameIdx].getTile( 0 ).setLeftTopXInFrame( 0 );
      context[frameIdx].getTile( 0 ).setLeftTopYInFrame( 0 );
      frameWidth  = context[frameIdx].getTile( 0 ).getWidth();
      frameHeight = context[frameIdx].getTile( 0 ).getHeight();

      auto& incomingTiles = context[frameIdx].getTiles();

      size_t tileIdx = 1;
      {
        // right
        incomingTiles[tileIdx].setLeftTopXInFrame( frameWidth );
        incomingTiles[tileIdx].setLeftTopYInFrame( 0 );
        frameWidth += incomingTiles[tileIdx].getWidth();
        // frameHeight = std::max (frameHeight, incomingTiles[tileIdx].getHeight());
      }

      tileIdx = 2;
      {
        // bottom left
        incomingTiles[tileIdx].setLeftTopXInFrame( 0 );
        incomingTiles[tileIdx].setLeftTopYInFrame( frameHeight );

        // frameWidth   = std::max ( frameWidth, incomingTiles[tileIdx].getWidth());
        frameHeight += incomingTiles[tileIdx].getHeight();
        frameHeight = std::max( frameHeight, incomingTiles[1].getHeight() );
      }

      if ( params_.losslessGeo_ || params_.lossyRawPointsPatch_ ) {
        tileIdx = 3;
        // bottom
        incomingTiles[tileIdx].setLeftTopXInFrame( 0 );
        if ( !params_.useRawPointsSeparateVideo_ ) {
          incomingTiles[tileIdx].setLeftTopYInFrame( frameHeight );
          frameWidth = std::max( frameWidth, incomingTiles[tileIdx].getWidth() );
          frameHeight += incomingTiles[tileIdx].getHeight();
        }
      }
      // copying to framesForPatchSegmentation
      auto& outputFrame = context[frameIdx].getTitleFrameContext();
      outputFrame.setWidth( frameWidth );
      outputFrame.setHeight( frameHeight );

      auto& outputFrameOccupanctMap = outputFrame.getOccupancyMap();
      outputFrameOccupanctMap.resize( frameWidth * frameHeight );
      outputFrame.getPatches().clear();
      for ( size_t tileIdx = 0; tileIdx < context[frameIdx].getNumTilesInAtlasFrame(); tileIdx++ ) {
        auto& inputTile          = context[frameIdx].getTile( tileIdx );
        auto& outputFramePatches = outputFrame.getPatches();
        auto& inputTilePatches   = inputTile.getPatches();
        for ( size_t patchIdx = 0; patchIdx < inputTilePatches.size(); patchIdx++ ) {
          outputFramePatches.push_back( inputTilePatches[patchIdx] );
          outputFramePatches[outputFramePatches.size() - 1].getU0() +=
              inputTile.getLeftTopXInFrame() / params_.occupancyResolution_;
          outputFramePatches[outputFramePatches.size() - 1].getV0() +=
              inputTile.getLeftTopYInFrame() / params_.occupancyResolution_;
        }
        // raw
        auto& outputFrameRawPatches = outputFrame.getRawPointsPatches();
        auto& inputTileRawPatches   = inputTile.getRawPointsPatches();
        outputFrameRawPatches.clear();
        for ( size_t patchIdx = 0; patchIdx < inputTileRawPatches.size(); patchIdx++ ) {
          outputFrameRawPatches.push_back( inputTileRawPatches[patchIdx] );
          if ( !params_.useRawPointsSeparateVideo_ ) {
            outputFrameRawPatches[outputFrameRawPatches.size() - 1].u0_ +=
                inputTile.getLeftTopXInFrame() / params_.occupancyResolution_;
            outputFrameRawPatches[outputFrameRawPatches.size() - 1].v0_ +=
                inputTile.getLeftTopYInFrame() / params_.occupancyResolution_;
          }
        }

        // eom
        auto& outputFrameEomPatches = outputFrame.getEomPatches();
        auto& inputTileEomPatches   = inputTile.getEomPatches();
        for ( size_t patchIdx = 0; patchIdx < inputTileEomPatches.size(); patchIdx++ ) {
          outputFrameEomPatches.push_back( inputTileEomPatches[patchIdx] );
          outputFrameEomPatches[outputFrameEomPatches.size() - 1].u0_ +=
              inputTile.getLeftTopXInFrame() / params_.occupancyResolution_;
          outputFrameEomPatches[outputFrameEomPatches.size() - 1].v0_ +=
              inputTile.getLeftTopYInFrame() / params_.occupancyResolution_;
        }

      }  // tileIdx

      context[frameIdx].updatePartitionInfoPerFrame( frameIdx, frameWidth, frameHeight, params_.numMaxTilePerFrame_,
                                                     true, params_.tilePartitionWidth_, params_.tilePartitionHeight_ );
#if 1
      for ( size_t ti = 0; ti < context[frameIdx].getNumTilesInAtlasFrame(); ti++ ) {
        if ( incomingTiles[ti].getPatches().size() != 0 )
          printf( "placeTiles frame %zu tile %zu: start %zu,%zu size %zux%zu\n", frameIdx, ti,
                  incomingTiles[ti].getLeftTopXInFrame(), incomingTiles[ti].getLeftTopYInFrame(),
                  incomingTiles[ti].getWidth(), incomingTiles[ti].getHeight() );
        else {
          printf( "placeTiles frame %zu tile %zu (patchEmpty) auxVideo %d: start %zu,%zu size %zux%zu\n", frameIdx, ti,
                  incomingTiles[ti].getUseRawPointsSeparateVideo(), incomingTiles[ti].getLeftTopXInFrame(),
                  incomingTiles[ti].getLeftTopYInFrame(), incomingTiles[ti].getWidth(), incomingTiles[ti].getHeight() );
        }
      }
#endif
    }  // frameIdx=0; frameIdx<context.getFrameCount(); frameIdx++)

    resizeGeometryVideo( context, params_.videoEncoderOccupancyCodecId_ );  // setAtalsWidth, Height
  }
}

void PCCEncoder::replaceFrameContext( PCCContext& context ) {
  for ( size_t frameIdx = 0; frameIdx < context.size(); frameIdx++ ) {
    if ( context[frameIdx].getNumTilesInAtlasFrame() == 1 ) continue;
    auto& frame = context[frameIdx].getTitleFrameContext();
    frame.setLeftTopXInFrame( 0 );
    frame.setLeftTopYInFrame( 0 );
    frame.setTileIndex( 255 );
    frame.getPatches().clear();

    size_t tileCount = context[frameIdx].getNumTilesInAtlasFrame();
    if ( params_.losslessGeo_ ) tileCount -= 1;
    for ( size_t tileIdx = 0; tileIdx < tileCount; tileIdx++ ) {
      auto& tile = context[frameIdx].getTile( tileIdx );
      for ( auto patch : tile.getPatches() ) {
        patch.getU0() += tile.getLeftTopXInFrame() / params_.occupancyResolution_;
        patch.getV0() += tile.getLeftTopYInFrame() / params_.occupancyResolution_;
        patch.getU1() += tile.getLeftTopXInFrame();
        patch.getV1() += tile.getLeftTopYInFrame();
        frame.getPatches().push_back( patch );
      }
    }
    // eom
    // raw
  }  // frame
}

void PCCEncoder::pointLocalReconstructionSearch( PCCContext&                          context,
                                                 PCCFrameContext&                     frame,
                                                 const std::vector<PCCVideoGeometry>& videoMultiple,
                                                 const GeneratePointCloudParameters&  params ) {
  auto& patches         = frame.getPatches();
  auto& blockToPatch    = frame.getBlockToPatch();
  auto& occupancyMapOrg = frame.getOccupancyMap();

  std::vector<uint32_t> occupancyMap;
  occupancyMap.resize( occupancyMapOrg.size(), 0 );
  for ( size_t i = 0; i < occupancyMapOrg.size(); i++ ) {
    occupancyMap[i] = static_cast<unsigned int>( occupancyMapOrg[i] >= 1 );
  }
  const size_t width              = frame.getWidth();
  const size_t height             = frame.getHeight();
  const size_t blockToPatchWidth  = width / params_.occupancyResolution_;
  const size_t blockToPatchHeight = height / params_.occupancyResolution_;
  const size_t blockSize0         = params_.occupancyResolution_ / params_.occupancyPrecision_;

  for ( size_t v0 = 0; v0 < blockToPatchHeight; ++v0 ) {
    for ( size_t u0 = 0; u0 < blockToPatchWidth; ++u0 ) {
      for ( size_t v1 = 0; v1 < blockSize0; ++v1 ) {
        const size_t v2 = v0 * params_.occupancyResolution_ + v1 * params_.occupancyPrecision_;
        for ( size_t u1 = 0; u1 < blockSize0; ++u1 ) {
          const size_t u2     = u0 * params_.occupancyResolution_ + u1 * params_.occupancyPrecision_;
          bool         isFull = false;
          for ( size_t v3 = 0; v3 < params_.occupancyPrecision_ && !isFull; ++v3 ) {
            for ( size_t u3 = 0; u3 < params_.occupancyPrecision_ && !isFull; ++u3 ) {
              isFull |= occupancyMap[( v2 + v3 ) * width + u2 + u3] == 1;
            }
          }
          for ( size_t v3 = 0; v3 < params_.occupancyPrecision_; ++v3 ) {
            for ( size_t u3 = 0; u3 < params_.occupancyPrecision_; ++u3 ) {
              occupancyMap[( v2 + v3 ) * width + u2 + u3] = static_cast<unsigned int>( isFull );
            }
          }
        }
      }
    }
  }
  size_t frameIndex;
  if ( params.multipleStreams_ ) {
    frameIndex = frame.getFrameIndex();
    if ( videoMultiple[0].getFrameCount() < ( frameIndex + 1 ) ) { return; }
  } else {
    frameIndex = frame.getFrameIndex() * ( params.mapCountMinus1_ + 1 );
    if ( videoMultiple[0].getFrameCount() < ( frameIndex + ( params.mapCountMinus1_ + 1 ) ) ) { return; }
  }
  const size_t patchCount           = patches.size();
  size_t       nbOfOptimizationMode = context.getPointLocalReconstructionModeNumber();
  const size_t imageWidth           = videoMultiple[0].getWidth();
  const size_t imageHeight          = videoMultiple[0].getHeight();

  for ( size_t patchIndex = 0; patchIndex < patchCount; ++patchIndex ) {
    const size_t  patchIndexPlusOne = patchIndex + 1;
    auto&         patch             = patches[patchIndex];
    const size_t& patchSize         = patch.getSizeU0() * patch.getSizeV0();
    if ( patchSize == 1 || patchSize <= params_.patchSize_ ) {
      patch.getPointLocalReconstructionLevel()     = 1;
      auto&                     srcPointCloudPatch = frame.getSrcPointCloudByPatch( patch.getOriginalIndex() );
      std::vector<PCCPointSet3> reconstruct;
      std::vector<float>        distance;
      reconstruct.resize( nbOfOptimizationMode );
      distance.resize( nbOfOptimizationMode );
      size_t optimizationIndex    = 0;
      size_t optimizationIndexMin = 0;
      for ( size_t i = 0; i < nbOfOptimizationMode; i++ ) {
        auto& mode = context.getPointLocalReconstructionMode( i );
        for ( size_t v0 = 0; v0 < patch.getSizeV0(); ++v0 ) {
          for ( size_t u0 = 0; u0 < patch.getSizeU0(); ++u0 ) {
            const size_t blockIndex = patch.patchBlock2CanvasBlock( u0, v0, blockToPatchWidth, blockToPatchHeight );
            if ( blockToPatch[blockIndex] == patchIndexPlusOne ) {
              for ( size_t v1 = 0; v1 < patch.getOccupancyResolution(); ++v1 ) {
                const size_t v = v0 * patch.getOccupancyResolution() + v1;
                for ( size_t u1 = 0; u1 < patch.getOccupancyResolution(); ++u1 ) {
                  const size_t u = u0 * patch.getOccupancyResolution() + u1;
                  size_t       x;
                  size_t       y;
                  const bool   occupancy = occupancyMap[patch.patch2Canvas( u, v, imageWidth, imageHeight, x, y )] != 0;
                  if ( !occupancy ) { continue; }
                  auto createdPoints = generatePoints( params, frame, videoMultiple, frameIndex, patchIndex, u, v, x, y,
                                                       mode.interpolate_, mode.filling_, mode.minD1_, mode.neighbor_ );
                  if ( !createdPoints.empty() ) {
                    for ( const auto& createdPoint : createdPoints ) {
                      reconstruct[optimizationIndex].addPoint( createdPoint );
                    }
                  }
                }
              }
            }
          }
        }
        float distancePSrcRec;
        float distancePRecSrc;
        srcPointCloudPatch.distanceGeo( reconstruct[optimizationIndex], distancePSrcRec, distancePRecSrc );
        distance[optimizationIndex] = ( std::max )( distancePSrcRec, distancePRecSrc );
        if ( optimizationIndex == 0 || distance[optimizationIndexMin] > distance[optimizationIndex] ) {
          optimizationIndexMin                    = optimizationIndex;
          patch.getPointLocalReconstructionMode() = optimizationIndexMin;
        }
        optimizationIndex++;
      }
    } else {
      patch.getPointLocalReconstructionLevel() = 0;
      for ( size_t v0 = 0; v0 < patch.getSizeV0(); ++v0 ) {
        for ( size_t u0 = 0; u0 < patch.getSizeU0(); ++u0 ) {
          patch.getPointLocalReconstructionMode( u0, v0 ) = 0;

          const size_t blockIndex = patch.patchBlock2CanvasBlock( u0, v0, blockToPatchWidth, blockToPatchHeight );
          if ( blockToPatch[blockIndex] == patchIndexPlusOne ) {
            auto&        srcPointCloudPatch = frame.getSrcPointCloudByPatch( patch.getOriginalIndex() );
            PCCPointSet3 blockSrcPointCloud;
            const size_t xMin = u0 * patch.getOccupancyResolution() + patch.getU1();
            const size_t yMin = v0 * patch.getOccupancyResolution() + patch.getV1();
            for ( size_t i = 0; i < srcPointCloudPatch.getPointCount(); i++ ) {
              if ( xMin <= srcPointCloudPatch[i][patch.getTangentAxis()] &&
                   srcPointCloudPatch[i][patch.getTangentAxis()] < xMin + patch.getOccupancyResolution() &&
                   yMin <= srcPointCloudPatch[i][patch.getBitangentAxis()] &&
                   srcPointCloudPatch[i][patch.getBitangentAxis()] < yMin + patch.getOccupancyResolution() ) {
                blockSrcPointCloud.addPoint( srcPointCloudPatch[i] );
              }
            }
            std::vector<PCCPointSet3> reconstruct;
            std::vector<float>        distance;
            reconstruct.resize( nbOfOptimizationMode );
            distance.resize( nbOfOptimizationMode );
            size_t optimizationIndex    = 0;
            size_t optimizationIndexMin = 0;
            for ( size_t i = 0; i < nbOfOptimizationMode; i++ ) {
              auto& mode = context.getPointLocalReconstructionMode( i );
              for ( size_t v1 = 0; v1 < patch.getOccupancyResolution(); ++v1 ) {
                const size_t v = v0 * patch.getOccupancyResolution() + v1;
                for ( size_t u1 = 0; u1 < patch.getOccupancyResolution(); ++u1 ) {
                  const size_t u = u0 * patch.getOccupancyResolution() + u1;
                  size_t       x;
                  size_t       y;
                  const bool   occupancy = occupancyMap[patch.patch2Canvas( u, v, imageWidth, imageHeight, x, y )] != 0;
                  if ( !occupancy ) { continue; }
                  auto createdPoints = generatePoints( params, frame, videoMultiple, frameIndex, patchIndex, u, v, x, y,
                                                       mode.interpolate_, mode.filling_, mode.minD1_, mode.neighbor_ );
                  if ( !createdPoints.empty() ) {
                    for ( const auto& createdPoint : createdPoints ) {
                      if ( patch.getAxisOfAdditionalPlane() == 0 ) {
                        reconstruct[optimizationIndex].addPoint( createdPoint );
                      } else {
                        PCCVector3D tmp;
                        PCCPatch::InverseRotatePosition45DegreeOnAxis( patch.getAxisOfAdditionalPlane(),
                                                                       params.geometryBitDepth3D_, createdPoint, tmp );
                        reconstruct[optimizationIndex].addPoint( tmp );
                      }
                    }
                  }
                }
              }
              float distancePSrcRec;
              float distancePRecSrc;
              blockSrcPointCloud.distanceGeo( reconstruct[optimizationIndex], distancePSrcRec, distancePRecSrc );
              distance[optimizationIndex] = ( std::max )( distancePSrcRec, distancePRecSrc );
              if ( optimizationIndex == 0 || distance[optimizationIndexMin] > distance[optimizationIndex] ) {
                optimizationIndexMin                            = optimizationIndex;
                patch.getPointLocalReconstructionMode( u0, v0 ) = optimizationIndexMin;
              }
              optimizationIndex++;
            }
          }  // if block is used
        }
      }
    }
  }  // patch
}

bool PCCEncoder::resizeGeometryVideo( PCCContext& context, PCCCodecId codecId ) {
  size_t maxWidth  = 0;
  size_t maxHeight = 0;
  for ( auto& frame : context.getFrames() ) {
    maxWidth  = ( std::max )( maxWidth, frame.getTitleFrameContext().getWidth() );
    maxHeight = ( std::max )( maxHeight, frame.getTitleFrameContext().getHeight() );
  }
  maxWidth  = ( std::max )( maxWidth, params_.minimumImageWidth_ );
  maxHeight = ( std::max )( maxHeight, params_.minimumImageHeight_ );
  maxWidth  = std::ceil( (double)maxWidth / 64.0 ) * 64;
  maxHeight = std::ceil( (double)maxHeight / 64.0 ) * 64;
#ifdef USE_JMAPP_VIDEO_CODEC
  if ( codecId == JMAPP ) {
    int ResMultiPre_occupancyResolution = params_.occupancyResolution_ * params_.occupancyPrecision_;
    if ( maxHeight % ResMultiPre_occupancyResolution != 0 ) {
      std::cout << "maxHeight % ResMultiPre_occupancyResolution != 0 " << std::endl;
      std::cout << "original maxHeight = " << maxHeight << std::endl;
      int rem   = maxHeight % ResMultiPre_occupancyResolution;
      maxHeight = maxHeight + ( ResMultiPre_occupancyResolution - rem );
      std::cout << "final maxHeight = " << maxHeight << std::endl;
    }
    if ( maxWidth % ResMultiPre_occupancyResolution != 0 ) {
      std::cout << "maxWidth % ResMultiPre_occupancyResolution != 0 " << std::endl;
      std::cout << "original maxWidth = " << maxWidth << std::endl;
      int rem  = maxWidth % ResMultiPre_occupancyResolution;
      maxWidth = maxWidth + ( ResMultiPre_occupancyResolution - rem );
      std::cout << "final maxHeight = " << maxHeight << std::endl;
    }
  }
#endif
  for ( auto& frame : context.getFrames() ) {
    if ( params_.tileSegmentationType_ == 1 ) {
      auto& partitionToTile = frame.getPartitionToTileMap();
      int   packingSize     = partitionToTile.size();
      int   tileWidth       = maxWidth / ( params_.numTilesHor_ * params_.occupancyResolution_ );
      int   tileHeight      = params_.tileHeightToWidthRatio_ * tileWidth;
      int   numTilesVert    = ( ( maxHeight + ( tileHeight * params_.occupancyResolution_ - 1 ) ) /
                           ( tileHeight * params_.occupancyResolution_ ) );
      int   newSize         = params_.numTilesHor_ * numTilesVert;
      partitionToTile.resize( newSize );
      for ( int i = packingSize; i < newSize; i++ ) partitionToTile[i] = -1;  // new added tiles have no ownership
    }
    frame.getTitleFrameContext().getWidth()  = maxWidth;
    frame.getTitleFrameContext().getHeight() = maxHeight;
    frame.getTitleFrameContext().getOccupancyMap().resize( ( maxWidth / params_.occupancyResolution_ ) *
                                                           ( maxHeight / params_.occupancyResolution_ ) );
    frame.setAtlasFrameWidth( maxWidth );
    frame.setAtlasFrameHeight( maxHeight );
    
    if( frame.getNumTilesInAtlasFrame() == 1 ){
      frame.setPartitionWidth(maxWidth, 0);
      frame.setPartitionHeight(maxHeight, 0);
    }
  }
  return true;
}
bool PCCEncoder::resizeTileGeometryVideo( PCCContext& context,
                                          size_t      tileIndex,
                                          size_t      frameWidth,
                                          size_t      frameHeight,
                                          int         firstFrame,
                                          int         lastFramePlus1 ) {
  size_t maxWidth = 0, maxHeight = 0;
  size_t startFrame = firstFrame < 0 ? 0 : firstFrame;
  size_t endFrame   = lastFramePlus1 < 0 ? context.size() : lastFramePlus1;
  for ( size_t frameIdx = startFrame; frameIdx < endFrame; frameIdx++ ) {
    auto& tile = context[frameIdx].getTile( tileIndex );
    maxWidth   = ( std::max )( maxWidth, tile.getWidth() );
    maxHeight  = ( std::max )( maxHeight, tile.getHeight() );
    if ( params_.tileSegmentationType_ != 0 ) {
      double partitionWidth  = (double)context[frameIdx].getPartitionWidth(0);
      double partitionHeight = (double)context[frameIdx].getPartitionHeight(0);
      maxWidth               = std::ceil( (double)maxWidth / partitionWidth ) * partitionWidth;
      maxHeight              = std::ceil( (double)maxHeight / partitionHeight ) * partitionHeight;
    }

  }  // frame
  if ( params_.tileSegmentationType_ == 0 ) {
    maxWidth  = ( std::max )( maxWidth, frameWidth );
    maxHeight = ( std::max )( maxHeight, frameHeight );
  } else {
    maxWidth  = std::ceil( (double)maxWidth / 64.0 ) * 64;
    maxHeight = std::ceil( (double)maxHeight / 64.0 ) * 64;
  }
  for ( size_t frameIdx = startFrame; frameIdx < endFrame; frameIdx++ ) {
    auto& tile       = context[frameIdx].getTile( tileIndex );
    tile.getWidth()  = maxWidth;
    tile.getHeight() = maxHeight;
    tile.getOccupancyMap().resize(
        ( maxWidth / params_.occupancyResolution_ ) * ( maxHeight / params_.occupancyResolution_ ), 0 );
    if(params_.tileSegmentationType_==0){
      context[frameIdx].setPartitionWidth(maxWidth, 0);
      context[frameIdx].setPartitionHeight(maxHeight, 0);
    }
  }  // frame
  return true;
}

size_t PCCEncoder::segmentSequence( PCCContext& context, std::vector<std::pair<size_t, size_t>>& framesInAFPS ) {
  size_t maxNumTiles = context[0].getNumTilesInAtlasFrame();
  framesInAFPS.resize( 1 );
  framesInAFPS[0].first  = 0;
  framesInAFPS[0].second = 0;
  for ( size_t frameIdx = 1; frameIdx < context.size(); frameIdx++ ) {
    bool bSameTileStruct =
        context[frameIdx].getPartitionToTileMap().size() == context[frameIdx - 1].getPartitionToTileMap().size();
    if ( bSameTileStruct ) {
      for ( size_t i = 0; i < context[frameIdx].getPartitionToTileMap().size(); i++ ) {
        bSameTileStruct &=
            ( context[frameIdx].getPartitionToTileMap()[i] == context[frameIdx - 1].getPartitionToTileMap()[i] );
        if ( !bSameTileStruct ) break;
      }
    }
    if ( bSameTileStruct ) {
      framesInAFPS[framesInAFPS.size() - 1].second++;
    } else {
      std::pair<size_t, size_t> newpair;
      newpair.first  = frameIdx;
      newpair.second = frameIdx;
      framesInAFPS.push_back( newpair );
    }
    maxNumTiles = std::max( maxNumTiles, context[frameIdx].getNumTilesInAtlasFrame() );
  }
  framesInAFPS[framesInAFPS.size() - 1].second = context.size() - 1;
  return maxNumTiles;
}
bool PCCEncoder::relocateTileGeometryVideo( PCCContext&                             context,
                                            std::vector<std::pair<size_t, size_t>>& framesInAFPS ) {
  for ( size_t segIdx = 0; segIdx < framesInAFPS.size(); segIdx++ ) {
    size_t firstFrame    = framesInAFPS[segIdx].first;
    size_t lastFrame     = framesInAFPS[segIdx].second;
    size_t numTilesInSeg = context[firstFrame].getNumTilesInAtlasFrame();
    if ( ( params_.losslessGeo_ || params_.lossyRawPointsPatch_ ) && !params_.useRawPointsSeparateVideo_ )
      numTilesInSeg -= 1;
    for ( size_t frameIdx = firstFrame; frameIdx < lastFrame + 1; frameIdx++ ) {
      size_t frameWidth  = context[frameIdx].getAtlasFrameWidth();
      size_t frameHeight = context[frameIdx].getTile( 0 ).getHeight();
      context[frameIdx].getTile( 0 ).setLeftTopXInFrame( 0 );
      context[frameIdx].getTile( 0 ).setLeftTopYInFrame( 0 );
      // checking 2 position : Right, next (partition) line
      for ( size_t tileIndex = 1; tileIndex < numTilesInSeg; tileIndex++ ) {
        auto& tile         = context[frameIdx].getTile( tileIndex );
        auto& tilePrev     = context[frameIdx].getTile( tileIndex - 1 );
        int   tempLeftTopX = tilePrev.getLeftTopXInFrame() + tilePrev.getWidth();
        int   tempLeftTopY = tilePrev.getLeftTopYInFrame() + tilePrev.getHeight();
        if ( tempLeftTopX + tile.getWidth() <= frameWidth ) {
          tempLeftTopY = tilePrev.getLeftTopYInFrame();
        } else {
          tempLeftTopX = 0;
        }
        size_t curLeftTopXinBlock = tile.getLeftTopXInFrame() / params_.occupancyResolution_;
        size_t curLeftTopYinBlock = tile.getLeftTopYInFrame() / params_.occupancyResolution_;
        size_t upLeftTopXinBlock  = tempLeftTopX / params_.occupancyResolution_;
        size_t upLeftTopYinBlock  = tempLeftTopY / params_.occupancyResolution_;
        tile.setLeftTopXInFrame( tempLeftTopX );
        tile.setLeftTopYInFrame( tempLeftTopY );
        assert( tempLeftTopX + tile.getWidth() <= frameWidth );
        frameHeight = std::max( frameHeight, (size_t)tempLeftTopY + tile.getHeight() );
      }

      context[frameIdx].setAtlasFrameWidth( frameWidth );
      context[frameIdx].setAtlasFrameHeight( frameHeight );
      size_t partitionWidthIn64 = ( context[frameIdx].getAtlasFrameWidth() / ( 64 * params_.numTilesHor_ ) );
      size_t partitionHeightIn64 =
          ( params_.tileHeightToWidthRatio_ * context[frameIdx].getAtlasFrameWidth() / ( 64 * params_.numTilesHor_ ) );
      bool bSinglePartitionPerTile = false;
      bool uniformPartitionSpacing = true;
      context[frameIdx].updatePartitionInfoPerFrame(
          frameIdx, frameWidth, frameHeight, context[firstFrame].getNumTilesInAtlasFrame(), uniformPartitionSpacing,
          partitionWidthIn64, partitionHeightIn64, -1, -1, bSinglePartitionPerTile, false );
    }
  }  // segIdx
  return true;
}

bool PCCEncoder::placeEomPatchInTile( PCCContext& context, std::vector<std::pair<size_t, size_t>>& framesInAFPS ) {
  for ( size_t segIdx = 0; segIdx < framesInAFPS.size(); segIdx++ ) {
    size_t firstFrame    = framesInAFPS[segIdx].first;
    size_t lastFrame     = framesInAFPS[segIdx].second;
    size_t numTilesInSeg = context[firstFrame].getNumTilesInAtlasFrame();
    if ( ( params_.losslessGeo_ || params_.lossyRawPointsPatch_ ) && !params_.useRawPointsSeparateVideo_ )
      numTilesInSeg -= 1;

    // find the last line of occupancy over frames
    std::vector<size_t> maxOccupiedHeightInBlock;
    maxOccupiedHeightInBlock.resize( numTilesInSeg, 0 );
    for ( size_t tileIdx = 0; tileIdx < numTilesInSeg; tileIdx++ ) {
      maxOccupiedHeightInBlock[tileIdx] = 0;
      for ( size_t frameIdx = firstFrame; frameIdx < lastFrame + 1; frameIdx++ ) {
        auto& tile = context[frameIdx].getTile( tileIdx );
        for ( auto& patch : tile.getPatches() ) {
          bool nonRotation = patch.getPatchOrientation() == PATCH_ORIENTATION_DEFAULT ||
                             patch.getPatchOrientation() == PATCH_ORIENTATION_ROT180 ||
                             patch.getPatchOrientation() == PATCH_ORIENTATION_MIRROR ||
                             patch.getPatchOrientation() == PATCH_ORIENTATION_MROT180;
          size_t patchImageHeightInBlock = nonRotation ? patch.getSizeV0() : patch.getSizeU0();
          maxOccupiedHeightInBlock[tileIdx] =
              std::max( maxOccupiedHeightInBlock[tileIdx], patch.getV0() + patchImageHeightInBlock );
        }
      }
    }

    for ( size_t frameIdx = firstFrame; frameIdx < lastFrame + 1; frameIdx++ ) {
      auto& atlasFrame = context[frameIdx].getTitleFrameContext();
      for ( size_t tileIdx = 0; tileIdx < numTilesInSeg; tileIdx++ ) {
        auto&             tile = context[frameIdx].getTile( tileIdx );
        std::vector<bool> occupancyMap;
        size_t occupancySizeU = context[frameIdx].getTile( tileIdx ).getWidth() / params_.occupancyResolution_;
        size_t occupancySizeV = 0;
        size_t tileWidth      = context[frameIdx].getTile( tileIdx ).getWidth();
        size_t tileHeight     = maxOccupiedHeightInBlock[tileIdx] * params_.occupancyResolution_;
        packEOMTexturePointsPatch( context[frameIdx].getTile( tileIdx ), occupancyMap, tileWidth, tileHeight,
                                   occupancySizeU, occupancySizeV, tileHeight );
        context[frameIdx].getTile( tileIdx ).setHeight( ceil( (double)tileHeight / 64.0 ) * 64 );
        // eom
        auto& outputFrameEomPatches = context[frameIdx].getTitleFrameContext().getEomPatches();
        auto& inputTileEomPatches   = context[frameIdx].getTile( tileIdx ).getEomPatches();
        outputFrameEomPatches.clear();
        for ( size_t patchIdx = 0; patchIdx < inputTileEomPatches.size(); patchIdx++ ) {
          outputFrameEomPatches.push_back( inputTileEomPatches[patchIdx] );
          outputFrameEomPatches[outputFrameEomPatches.size() - 1].u0_ +=
              context[frameIdx].getTile( tileIdx ).getLeftTopXInFrame() / params_.occupancyResolution_;
          outputFrameEomPatches[outputFrameEomPatches.size() - 1].v0_ +=
              context[frameIdx].getTile( tileIdx ).getLeftTopYInFrame() / params_.occupancyResolution_;
        }
#if 0
        std::cout << "\t(placeEomPatchInTile) frame "<<atlasFrame.getFrameIndex()<<", tile "<<tile.getTileIndex()<<"\teomPatchSize: "<<tile.getEomPatches().size()<<"\t";
        for(size_t ii=0; ii<tile.getEomPatches().size(); ii++)
          std::cout<<"\t"<<ii<<" starts:"<<tile.getEomPatches(ii).u0_<<"x"<<tile.getEomPatches(ii).v0_<<" size:"<<tile.getEomPatches(ii).sizeU_<<"x"<<tile.getEomPatches(ii).sizeV_<<"\n";
#endif
      }
    }  // frameIdx
    for ( size_t tileIdx = 0; tileIdx < numTilesInSeg; tileIdx++ ) {
      size_t initTileWidth  = context.getFrame( firstFrame ).getTile( tileIdx ).getWidth();
      size_t initTileHeight = context.getFrame( firstFrame ).getTile( tileIdx ).getHeight();
      resizeTileGeometryVideo( context, tileIdx, initTileWidth, initTileHeight, firstFrame, lastFrame + 1 );
    }
  }  // segIdx
  return true;
}
bool PCCEncoder::placeRawPatchTile( PCCContext& context, std::vector<std::pair<size_t, size_t>>& framesInAFPS ) {
  for ( size_t segIdx = 0; segIdx < framesInAFPS.size(); segIdx++ ) {
    size_t firstFrame    = framesInAFPS[segIdx].first;
    size_t lastFrame     = framesInAFPS[segIdx].second;
    size_t numTilesInSeg = context[firstFrame].getNumTilesInAtlasFrame();
    for ( size_t frameIdx = firstFrame; frameIdx < lastFrame + 1; frameIdx++ ) {
      auto& atlasFrame = context[frameIdx].getTitleFrameContext();
      auto& tile       = context[frameIdx].getTile( context[frameIdx].getNumTilesInAtlasFrame() - 1 );
      if ( tile.getNumberOfRawPointsPatches() == 0 ) continue;
      tile.setLeftTopXInFrame( 0 );
      tile.setLeftTopYInFrame( atlasFrame.getHeight() );
      tile.setWidth( std::max( size_t( 64 ), atlasFrame.getWidth() ) );
      size_t tileHeight = packRawPointsPatchSimple( tile, 0, 0 );
      tile.setHeight( std::ceil( (double)tileHeight / 64.0 ) * 64 );
      atlasFrame.setHeight( atlasFrame.getHeight() + tile.getHeight() );
      auto& atlasFrameRawPatches = atlasFrame.getRawPointsPatches();
      atlasFrameRawPatches.clear();
      for ( size_t patchIdx = 0; patchIdx < tile.getRawPointsPatches().size(); patchIdx++ ) {
        atlasFrameRawPatches.push_back( tile.getRawPointsPatch( patchIdx ) );
        atlasFrameRawPatches[patchIdx].u0_ += tile.getLeftTopXInFrame() / params_.occupancyResolution_;
        atlasFrameRawPatches[patchIdx].v0_ += tile.getLeftTopYInFrame() / params_.occupancyResolution_;
      }
      std::cout << "->frame " << atlasFrame.getFrameIndex() << " tile " << tile.getTileIndex()
                << "(raw patches only) start @" << tile.getLeftTopXInFrame() << "," << tile.getLeftTopYInFrame()
                << " size: " << tile.getWidth() << "x" << tile.getHeight() << " patchSize: " << tile.getPatches().size()
                << "\trawPatchSize: " << tile.getRawPointsPatches().size() << "\t";
      std::cout << " ImageSize " << atlasFrame.getWidth() << " x " << atlasFrame.getHeight() << "\n";
      std::cout << "copyToFrame:" << atlasFrameRawPatches.size() << "\n";
      for ( size_t ii = 0; ii < atlasFrameRawPatches.size(); ii++ ) {
        std::cout << "atlasFrameRawPatch[" << ii << "]:" << atlasFrameRawPatches[ii].u0_ << ", "
                  << atlasFrameRawPatches[ii].v0_ << "\t";
        std::cout << atlasFrameRawPatches[ii].sizeU_ << "x" << atlasFrameRawPatches[ii].sizeV_
                  << " #points: " << atlasFrameRawPatches[ii].getNumberOfRawPoints() << std::endl;
      }
      size_t partitionWidthIn64 = ( context[frameIdx].getAtlasFrameWidth() / ( 64 * params_.numTilesHor_ ) );
      size_t partitionHeightIn64 =
          ( params_.tileHeightToWidthRatio_ * context[frameIdx].getAtlasFrameWidth() / ( 64 * params_.numTilesHor_ ) );

      bool bSinglePartitionPerTile = false;
      bool uniformPartitionSpacing = true;
      context[frameIdx].updatePartitionInfoPerFrame(
          frameIdx, atlasFrame.getWidth(), atlasFrame.getHeight(), context[firstFrame].getNumTilesInAtlasFrame(),
          uniformPartitionSpacing, partitionWidthIn64, partitionHeightIn64, -1, -1, bSinglePartitionPerTile, false );

    }  // frameIdx
  }    // segIdx
  return true;
}

void PCCEncoder::markRawPatchLocationOccupancyMapVideo( PCCContext& context ) {
  auto& videoOccupancyMap = context.getVideoOccupancyMap();
  for ( size_t f = 0; f < context.size(); f++ ) {
    markRawPatchLocation( context[f].getTitleFrameContext(), videoOccupancyMap.getFrame( f ) );
  }
}

void PCCEncoder::markRawPatchLocation( PCCFrameContext& frame, PCCImageOccupancyMap& imageOccupancyMap ) {
  if ( !frame.getUseRawPointsSeparateVideo() ) {
    size_t videoWidth               = frame.getWidth();
    size_t videoHeight              = frame.getHeight();
    size_t numberOfRawPointsPatches = frame.getNumberOfRawPointsPatches();
    for ( int i = 0; i < numberOfRawPointsPatches; i++ ) {
      auto&        rawPointsPatch = frame.getRawPointsPatch( i );
      const size_t v0             = rawPointsPatch.v0_ * rawPointsPatch.occupancyResolution_;
      const size_t u0             = rawPointsPatch.u0_ * rawPointsPatch.occupancyResolution_;
      if ( rawPointsPatch.sizeX() != 0u ) {
        for ( size_t v = 0; v < rawPointsPatch.sizeV_; ++v ) {
          for ( size_t u = 0; u < rawPointsPatch.sizeU_; ++u ) {
            const size_t p = v * rawPointsPatch.sizeU_ + u;
            if ( p < rawPointsPatch.getNumberOfRawPoints() * 3 ) {
              const size_t x = ( u0 + u );
              const size_t y = ( v0 + v );
              if ( !params_.lossyRawPointsPatch_ ) {
                if ( x >= videoWidth || y >= videoHeight ) {
                  std::cout << "\t\tframe [" << frame.getFrameIndex() << "] out of image :" << x << "," << y << "("
                            << x + y * videoWidth << ") vs occupancyMap size :" << videoWidth << "x" << videoHeight
                            << std::endl;
                  std::cout << "rawpatch[" << i << "] #point:" << rawPointsPatch.getNumberOfRawPoints()
                            << " size: " << rawPointsPatch.sizeU_ << "x" << rawPointsPatch.sizeV_ << "\tat " << u0
                            << ", " << v0 << std::endl;
                  exit( 0 );
                }
                assert( x < videoWidth && y < videoHeight );
                imageOccupancyMap.setValue( 0, x, y, 1 );
              }
              frame.getOccupancyMap()[x + y * videoWidth] = 1;
            }
          }
        }
      }
    }
  }
}

bool PCCEncoder::generateGeometryVideo( const PCCGroupOfFrames& sources, PCCContext& context ) {
  auto& videoGeometry         = context.getVideoGeometryMultiple()[0];
  auto& videoGeometryMultiple = context.getVideoGeometryMultiple();
  auto& videoOccupancyMap     = context.getVideoOccupancyMap();
  auto& frameInfos            = context.getFrames();

  for ( size_t i = 0; i < frameInfos.size(); i++ ) {
    auto& frame = frameInfos[i].getTitleFrameContext();
    if ( !params_.useRawPointsSeparateVideo_ && ( params_.losslessGeo_ || params_.lossyRawPointsPatch_ ) ) {
      markRawPatchLocation( frame, videoOccupancyMap.getFrame( i ) );
    }
    if ( params_.multipleStreams_ ) {
      const size_t geometryVideoSize = videoGeometryMultiple[0].getFrameCount();
      videoGeometryMultiple[0].resize( geometryVideoSize + 1 );
      videoGeometryMultiple[1].resize( geometryVideoSize + 1 );
      auto& frame0 = videoGeometryMultiple[0].getFrame( geometryVideoSize );
      generateIntraImage( frameInfos[i], 0, frame0 );
      auto& frame1 = videoGeometryMultiple[1].getFrame( geometryVideoSize );
      generateIntraImage( frameInfos[i], 1, frame1 );
      dilate3DPadding( sources[i], frameInfos[i], frame, frame0, videoOccupancyMap.getFrame( i ) );
      if ( params_.absoluteD1_ ) {
        dilate3DPadding( sources[i], frameInfos[i], frame, frame1, videoOccupancyMap.getFrame( i ) );
      }
    } else {
      const size_t geometryVideoSize = videoGeometry.getFrameCount();
      const size_t mapCount          = params_.mapCountMinus1_ + 1;
      videoGeometry.resize( geometryVideoSize + mapCount );
      if ( params_.singleMapPixelInterleaving_ ) {
        auto& frame1 = videoGeometry.getFrame( geometryVideoSize );
        generateIntraImage( frameInfos[i], 0, frame1 );
        dilate( frame, frame1 );
        PCCImageGeometry frame2;
        generateIntraImage( frameInfos[i], 1, frame2 );
        dilate3DPadding( sources[i], frameInfos[i], frame, frame2, videoOccupancyMap.getFrame( i ) );
        for ( size_t x = 0; x < frame1.getWidth(); x++ ) {
          for ( size_t y = 0; y < frame1.getHeight(); y++ ) {
            if ( ( x + y ) % 2 == 1 ) { frame1.setValue( 0, x, y, frame2.getValue( 0, x, y ) ); }
          }
        }
      } else {
        for ( size_t f = 0; f < mapCount; ++f ) {
          auto& geoImage = videoGeometry.getFrame( geometryVideoSize + f );
          generateIntraImage( frameInfos[i], f, geoImage );
          dilate3DPadding( sources[i], frameInfos[i], frame, geoImage, videoOccupancyMap.getFrame( i ) );
        }
      }
    }
    // Group dilation in Geometry
    if ( params_.groupDilation_ && params_.absoluteD1_ && params_.mapCountMinus1_ > 0 ) {
      dilateGroupGeometryVideo( context, frame, i );
    }
  }  // frame
  return true;
}

template <typename T>
void PCCEncoder::dilate( PCCFrameContext& frame, PCCImage<T, 3>& image, const PCCImage<T, 3>* reference ) {
  auto          occupancyMapTemp         = frame.getOccupancyMap();
  const size_t  pixelBlockCount          = params_.occupancyResolution_ * params_.occupancyResolution_;
  const size_t  occupancyMapSizeU        = image.getWidth() / params_.occupancyResolution_;
  const size_t  occupancyMapSizeV        = image.getHeight() / params_.occupancyResolution_;
  const int64_t neighbors[4][2]          = {{0, -1}, {-1, 0}, {1, 0}, {0, 1}};
  const size_t  MAX_OCCUPANCY_RESOLUTION = 64;
  assert( params_.occupancyResolution_ <= MAX_OCCUPANCY_RESOLUTION );
  size_t              count[MAX_OCCUPANCY_RESOLUTION][MAX_OCCUPANCY_RESOLUTION];
  PCCVector3<int32_t> values[MAX_OCCUPANCY_RESOLUTION][MAX_OCCUPANCY_RESOLUTION];

  for ( size_t v1 = 0; v1 < occupancyMapSizeV; ++v1 ) {
    const int64_t v0 = v1 * params_.occupancyResolution_;
    for ( size_t u1 = 0; u1 < occupancyMapSizeU; ++u1 ) {
      const int64_t u0                = u1 * params_.occupancyResolution_;
      size_t        nonZeroPixelCount = 0;
      for ( size_t v2 = 0; v2 < params_.occupancyResolution_; ++v2 ) {
        for ( size_t u2 = 0; u2 < params_.occupancyResolution_; ++u2 ) {
          const int64_t x0 = u0 + u2;
          const int64_t y0 = v0 + v2;
          assert( x0 < int64_t( image.getWidth() ) && y0 < int64_t( image.getHeight() ) );
          const size_t location0 = y0 * image.getWidth() + x0;
          if ( params_.enhancedOccupancyMapCode_ ) {
            nonZeroPixelCount += ( occupancyMapTemp[location0] > 0 );
          } else {
            nonZeroPixelCount += ( occupancyMapTemp[location0] == 1 );
          }
        }
      }
      if ( !nonZeroPixelCount ) {
        if ( reference ) {
          for ( size_t v2 = 0; v2 < params_.occupancyResolution_; ++v2 ) {
            for ( size_t u2 = 0; u2 < params_.occupancyResolution_; ++u2 ) {
              const size_t x0 = u0 + u2;
              const size_t y0 = v0 + v2;
              image.setValue( 0, x0, y0, reference->getValue( 0, x0, y0 ) );
              image.setValue( 1, x0, y0, reference->getValue( 1, x0, y0 ) );
              image.setValue( 2, x0, y0, reference->getValue( 2, x0, y0 ) );
            }
          }
        } else if ( u1 > 0 ) {
          for ( size_t v2 = 0; v2 < params_.occupancyResolution_; ++v2 ) {
            for ( size_t u2 = 0; u2 < params_.occupancyResolution_; ++u2 ) {
              const size_t x0 = u0 + u2;
              const size_t y0 = v0 + v2;
              assert( x0 > 0 );
              const size_t x1 = x0 - 1;
              image.setValue( 0, x0, y0, image.getValue( 0, x1, y0 ) );
              image.setValue( 1, x0, y0, image.getValue( 1, x1, y0 ) );
              image.setValue( 2, x0, y0, image.getValue( 2, x1, y0 ) );
            }
          }
        } else if ( v1 > 0 ) {
          for ( size_t v2 = 0; v2 < params_.occupancyResolution_; ++v2 ) {
            for ( size_t u2 = 0; u2 < params_.occupancyResolution_; ++u2 ) {
              const size_t x0 = u0 + u2;
              const size_t y0 = v0 + v2;
              assert( y0 > 0 );
              const size_t y1 = y0 - 1;
              image.setValue( 0, x0, y0, image.getValue( 0, x0, y1 ) );
              image.setValue( 1, x0, y0, image.getValue( 1, x0, y1 ) );
              image.setValue( 2, x0, y0, image.getValue( 2, x0, y1 ) );
            }
          }
        }
        continue;
      }
      for ( size_t v2 = 0; v2 < params_.occupancyResolution_; ++v2 ) {
        for ( size_t u2 = 0; u2 < params_.occupancyResolution_; ++u2 ) {
          values[v2][u2] = 0;
          count[v2][u2]  = 0UL;
        }
      }
      uint32_t iteration = 1;
      while ( nonZeroPixelCount < pixelBlockCount ) {
        for ( size_t v2 = 0; v2 < params_.occupancyResolution_; ++v2 ) {
          for ( size_t u2 = 0; u2 < params_.occupancyResolution_; ++u2 ) {
            const int64_t x0 = u0 + u2;
            const int64_t y0 = v0 + v2;
            assert( x0 < int64_t( image.getWidth() ) && y0 < int64_t( image.getHeight() ) );
            const size_t location0 = y0 * image.getWidth() + x0;
            if ( occupancyMapTemp[location0] == iteration ) {
              for ( auto neighbor : neighbors ) {
                const int64_t x1        = x0 + neighbor[0];
                const int64_t y1        = y0 + neighbor[1];
                const size_t  location1 = y1 * image.getWidth() + x1;
                if ( x1 >= u0 && x1 < int64_t( u0 + params_.occupancyResolution_ ) && y1 >= v0 &&
                     y1 < int64_t( v0 + params_.occupancyResolution_ ) && occupancyMapTemp[location1] == 0 ) {
                  const int64_t u3 = u2 + neighbor[0];
                  const int64_t v3 = v2 + neighbor[1];
                  assert( u3 >= 0 && u3 < int64_t( params_.occupancyResolution_ ) );
                  assert( v3 >= 0 && v3 < int64_t( params_.occupancyResolution_ ) );
                  for ( size_t k = 0; k < 3; ++k ) { values[v3][u3][k] += image.getValue( k, x0, y0 ); }
                  ++count[v3][u3];
                }
              }
            }
          }
        }
        for ( size_t v2 = 0; v2 < params_.occupancyResolution_; ++v2 ) {
          for ( size_t u2 = 0; u2 < params_.occupancyResolution_; ++u2 ) {
            if ( count[v2][u2] ) {
              ++nonZeroPixelCount;
              const size_t x0             = u0 + u2;
              const size_t y0             = v0 + v2;
              const size_t location0      = y0 * image.getWidth() + x0;
              const size_t c              = count[v2][u2];
              const size_t c2             = c / 2;
              occupancyMapTemp[location0] = iteration + 1;
              for ( size_t k = 0; k < 3; ++k ) { image.setValue( k, x0, y0, T( ( values[v2][u2][k] + c2 ) / c ) ); }
              values[v2][u2] = 0;
              count[v2][u2]  = 0UL;
            }
          }
        }
        ++iteration;
      }
    }
  }
}

// 3D geometry padding
size_t PCCEncoder::adjustDepth3DPadding( size_t            x,
                                         size_t            y,
                                         uint16_t          mean_val,
                                         PCCImageGeometry& image,
                                         PCCKdTree&        kdtree,
                                         PCCFrameContext&  frame ) {
  auto&  blockToPatch = frame.getBlockToPatch();
  auto&  patches      = frame.getPatches();
  size_t block_addr   = ( y / params_.occupancyResolution_ ) * ( image.getWidth() / params_.occupancyResolution_ ) +
                      x / params_.occupancyResolution_;
  size_t patchIndex = blockToPatch[block_addr];
  auto&  patch      = patches[patchIndex - 1];
  size_t distance   = ( std::numeric_limits<int16_t>::max )();
  // testing the mean value
  PCCNNResult result_mean;
  PCCPoint3D  point_mean = patch.canvasTo3D( x, y, mean_val );
  kdtree.search( point_mean, 1, result_mean );
  const double dist2_mean = result_mean.dist( 0 );
  if ( dist2_mean < distance ) {
    image.setValue( 0, x, y, mean_val );
    image.setValue( 1, x, y, 0 );
    image.setValue( 2, x, y, 0 );
    distance = dist2_mean;
  }
  if ( distance != 0 ) {
    // the mean value does not belong to the point cloud, so let's search for a
    // nearby value and see if it is better
    // than the mean.
    size_t deltadepth = 8;
    if ( mean_val < deltadepth ) { deltadepth = mean_val; }
    if ( mean_val + deltadepth > frame.getMaxDepth() ) { deltadepth = frame.getMaxDepth() - mean_val; }
    for ( uint16_t depth = 1; depth < deltadepth; depth++ ) {
      PCCPoint3D point = patch.canvasTo3D( x, y, mean_val + depth );
      // now find the distance between the point and the original point cloud
      PCCNNResult result;
      kdtree.search( point, 1, result );
      const double dist2 = result.dist( 0 );
      if ( dist2 < distance ) {
        image.setValue( 0, x, y, mean_val + depth );
        image.setValue( 1, x, y, 0 );
        image.setValue( 2, x, y, 0 );
        distance = dist2;
      }
      PCCPoint3D point_neg = patch.canvasTo3D( x, y, mean_val - depth );
      // now find the distance between the point and the original point cloud
      PCCNNResult result_neg;
      kdtree.search( point_neg, 1, result_neg );
      const double dist2_neg = result_neg.dist( 0 );
      if ( dist2_neg < distance ) {
        image.setValue( 0, x, y, mean_val - depth );
        image.setValue( 1, x, y, 0 );
        image.setValue( 2, x, y, 0 );
        distance = dist2_neg;
      }
    }
  }
  return 1;
}

void PCCEncoder::dilate3DPadding( const PCCPointSet3&     source,
                                  PCCAtlasFrameContext&   frameInfo,
                                  PCCFrameContext&        frame,
                                  PCCImageGeometry&       image,
                                  PCCImageOccupancyMap&   occupancyMap,
                                  const PCCImageGeometry* reference ) {
  const size_t  pixelBlockCount          = params_.occupancyResolution_ * params_.occupancyResolution_;
  const size_t  occupancyMapSizeU        = image.getWidth() / params_.occupancyResolution_;
  const size_t  occupancyMapSizeV        = image.getHeight() / params_.occupancyResolution_;
  const int64_t neighbors[4][2]          = {{0, -1}, {-1, 0}, {1, 0}, {0, 1}};
  const size_t  MAX_OCCUPANCY_RESOLUTION = 64;
  assert( params_.occupancyResolution_ <= MAX_OCCUPANCY_RESOLUTION );
  size_t              count[MAX_OCCUPANCY_RESOLUTION][MAX_OCCUPANCY_RESOLUTION];
  PCCVector3<int32_t> values[MAX_OCCUPANCY_RESOLUTION][MAX_OCCUPANCY_RESOLUTION];

  std::vector<uint32_t> occupancyMapTemp;
  auto&                 occupancyMapOriginal = frame.getOccupancyMap();
  occupancyMapTemp.resize( image.getWidth() * image.getHeight(), 0 );
  PCCKdTree kdtree( source );
  // fill in positions that are added to the sequence, because of occupancyMap
  // video coding

  for ( size_t y_OM = 0; y_OM < occupancyMap.getHeight(); ++y_OM ) {
    for ( size_t x_OM = 0; x_OM < occupancyMap.getWidth(); ++x_OM ) {
      if ( occupancyMap.getValue( 0, x_OM, y_OM ) >= 1 ) {
        // this is an area that has active values, update the temporary
        // occupancy Map struture, and store the mean
        // value in this area
        uint16_t mean_val = 0;
        size_t   count    = 0;
        for ( size_t j = 0; j < params_.occupancyPrecision_; j++ ) {
          size_t y = y_OM * params_.occupancyPrecision_ + j;
          for ( size_t i = 0; i < params_.occupancyPrecision_; i++ ) {
            size_t x = x_OM * params_.occupancyPrecision_ + i;
            if ( occupancyMapOriginal[y * image.getWidth() + x] != 0 ) {
              mean_val += image.getValue( 0, x, y );
              count++;
            }
          }
        }
#if 1
        if ( count == 0 ) {
          printf( "dilate3DPadding %zu frame- %zux%zu, %zux%zu frame : (%zu,%zu) in OM (%zu,%zu)\n",
                  frame.getFrameIndex(), frame.getWidth(), frame.getHeight(), image.getWidth(), image.getHeight(), x_OM,
                  y_OM, x_OM * params_.occupancyPrecision_, y_OM * params_.occupancyPrecision_ );
          exit( 123 );
        }
#endif
        assert( count > 0 );

        mean_val /= count;
        // now fill in the missing positions with depth values searched in 3D
        // space
        for ( size_t j = 0; j < params_.occupancyPrecision_; j++ ) {
          size_t y = y_OM * params_.occupancyPrecision_ + j;
          for ( size_t i = 0; i < params_.occupancyPrecision_; i++ ) {
            size_t x = x_OM * params_.occupancyPrecision_ + i;
            // if depth value is undefined, this position will be added, find
            // the best value
            if ( occupancyMapOriginal[y * image.getWidth() + x] == 0 ) {
              // try to find the best value to approximate this new point to the
              // original point cloud
              // get the patch information
              if ( params_.geometryPadding_ == 1 ) {
                occupancyMapTemp[y * image.getWidth() + x] =
                    adjustDepth3DPadding( x, y, mean_val, image, kdtree, frame );
              } else {
                occupancyMapTemp[y * image.getWidth() + x] = 0;
              }
            } else {
              occupancyMapTemp[y * image.getWidth() + x] = 1;
            }
          }
        }
      }
    }
  }

  // now continue adding the pixels with the previous dilation approach
  for ( size_t v1 = 0; v1 < occupancyMapSizeV; ++v1 ) {
    const int64_t v0 = v1 * params_.occupancyResolution_;
    for ( size_t u1 = 0; u1 < occupancyMapSizeU; ++u1 ) {
      const int64_t u0                = u1 * params_.occupancyResolution_;
      size_t        nonZeroPixelCount = 0;
      for ( size_t v2 = 0; v2 < params_.occupancyResolution_; ++v2 ) {
        for ( size_t u2 = 0; u2 < params_.occupancyResolution_; ++u2 ) {
          const int64_t x0 = u0 + u2;
          const int64_t y0 = v0 + v2;
          assert( x0 < int64_t( image.getWidth() ) && y0 < int64_t( image.getHeight() ) );
          const size_t location0 = y0 * image.getWidth() + x0;
          if ( params_.enhancedOccupancyMapCode_ ) {
            nonZeroPixelCount += static_cast<unsigned long long>( occupancyMapTemp[location0] > 0 );
          } else {
            nonZeroPixelCount += static_cast<unsigned long long>( occupancyMapTemp[location0] == 1 );
          }
        }
      }
      if ( nonZeroPixelCount == 0u ) {
        if ( reference != nullptr ) {
          for ( size_t v2 = 0; v2 < params_.occupancyResolution_; ++v2 ) {
            for ( size_t u2 = 0; u2 < params_.occupancyResolution_; ++u2 ) {
              const size_t x0 = u0 + u2;
              const size_t y0 = v0 + v2;
              image.setValue( 0, x0, y0, reference->getValue( 0, x0, y0 ) );
              image.setValue( 1, x0, y0, reference->getValue( 1, x0, y0 ) );
              image.setValue( 2, x0, y0, reference->getValue( 2, x0, y0 ) );
            }
          }
        } else if ( u1 > 0 ) {
          for ( size_t v2 = 0; v2 < params_.occupancyResolution_; ++v2 ) {
            for ( size_t u2 = 0; u2 < params_.occupancyResolution_; ++u2 ) {
              const size_t x0 = u0 + u2;
              const size_t y0 = v0 + v2;
              assert( x0 > 0 );
              const size_t x1 = x0 - 1;
              image.setValue( 0, x0, y0, image.getValue( 0, x1, y0 ) );
              image.setValue( 1, x0, y0, image.getValue( 1, x1, y0 ) );
              image.setValue( 2, x0, y0, image.getValue( 2, x1, y0 ) );
            }
          }
        } else if ( v1 > 0 ) {
          for ( size_t v2 = 0; v2 < params_.occupancyResolution_; ++v2 ) {
            for ( size_t u2 = 0; u2 < params_.occupancyResolution_; ++u2 ) {
              const size_t x0 = u0 + u2;
              const size_t y0 = v0 + v2;
              assert( y0 > 0 );
              const size_t y1 = y0 - 1;
              image.setValue( 0, x0, y0, image.getValue( 0, x0, y1 ) );
              image.setValue( 1, x0, y0, image.getValue( 1, x0, y1 ) );
              image.setValue( 2, x0, y0, image.getValue( 2, x0, y1 ) );
            }
          }
        }
        continue;
      }
      for ( size_t v2 = 0; v2 < params_.occupancyResolution_; ++v2 ) {
        for ( size_t u2 = 0; u2 < params_.occupancyResolution_; ++u2 ) {
          values[v2][u2] = 0;
          count[v2][u2]  = 0UL;
        }
      }
      uint32_t iteration = 1;
      while ( nonZeroPixelCount < pixelBlockCount ) {
        for ( size_t v2 = 0; v2 < params_.occupancyResolution_; ++v2 ) {
          for ( size_t u2 = 0; u2 < params_.occupancyResolution_; ++u2 ) {
            const int64_t x0 = u0 + u2;
            const int64_t y0 = v0 + v2;
            assert( x0 < int64_t( image.getWidth() ) && y0 < int64_t( image.getHeight() ) );
            const size_t location0 = y0 * image.getWidth() + x0;
            if ( occupancyMapTemp[location0] == iteration ) {
              for ( auto neighbor : neighbors ) {
                const int64_t x1        = x0 + neighbor[0];
                const int64_t y1        = y0 + neighbor[1];
                const size_t  location1 = y1 * image.getWidth() + x1;
                if ( x1 >= u0 && x1 < int64_t( u0 + params_.occupancyResolution_ ) && y1 >= v0 &&
                     y1 < int64_t( v0 + params_.occupancyResolution_ ) && occupancyMapTemp[location1] == 0 ) {
                  const int64_t u3 = u2 + neighbor[0];
                  const int64_t v3 = v2 + neighbor[1];
                  assert( u3 >= 0 && u3 < int64_t( params_.occupancyResolution_ ) );
                  assert( v3 >= 0 && v3 < int64_t( params_.occupancyResolution_ ) );
                  for ( size_t k = 0; k < 3; ++k ) { values[v3][u3][k] += image.getValue( k, x0, y0 ); }
                  ++count[v3][u3];
                }
              }
            }
          }
        }
        for ( size_t v2 = 0; v2 < params_.occupancyResolution_; ++v2 ) {
          for ( size_t u2 = 0; u2 < params_.occupancyResolution_; ++u2 ) {
            if ( count[v2][u2] != 0u ) {
              ++nonZeroPixelCount;
              const size_t x0             = u0 + u2;
              const size_t y0             = v0 + v2;
              const size_t location0      = y0 * image.getWidth() + x0;
              const size_t c              = count[v2][u2];
              const size_t c2             = c / 2;
              occupancyMapTemp[location0] = iteration + 1;
              for ( size_t k = 0; k < 3; ++k ) {
                image.setValue( k, x0, y0, uint16_t( ( values[v2][u2][k] + c2 ) / c ) );
              }
              values[v2][u2] = 0;
              count[v2][u2]  = 0UL;
            }
          }
        }
        ++iteration;
      }
    }
  }
}

/* harmonic background filling algorithm */
// interpolate using 5-point laplacian inpainting
template <typename T>
void PCCEncoder::dilateHarmonicBackgroundFill( PCCFrameContext& frame, PCCImage<T, 3>& image ) {
  auto                               occupancyMapTemp = frame.getOccupancyMap();
  int                                i                = 0;
  std::vector<PCCImage<T, 3>>        mipVec;
  std::vector<std::vector<uint32_t>> mipOccupancyMapVec;
  int                                miplev = 0;

  // create coarse image by dyadic sampling
  while ( true ) {
    mipVec.resize( mipVec.size() + 1 );
    mipOccupancyMapVec.resize( mipOccupancyMapVec.size() + 1 );
    if ( miplev > 0 ) {
      CreateCoarseLayer( mipVec[miplev - 1], mipVec[miplev], mipOccupancyMapVec[miplev - 1],
                         mipOccupancyMapVec[miplev] );
    } else {
      CreateCoarseLayer( image, mipVec[miplev], occupancyMapTemp, mipOccupancyMapVec[miplev] );
    }

    if ( mipVec[miplev].getWidth() <= 4 || mipVec[miplev].getHeight() <= 4 ) { break; }
    ++miplev;
  }
  miplev++;
  // push phase: inpaint laplacian
  regionFill( mipVec[miplev - 1], mipOccupancyMapVec[miplev - 1], mipVec[miplev - 1] );
  for ( i = miplev - 1; i >= 0; --i ) {
    if ( i > 0 ) {
      regionFill( mipVec[i - 1], mipOccupancyMapVec[i - 1], mipVec[i] );
    } else {
      regionFill( image, occupancyMapTemp, mipVec[i] );
    }
  }
}

template <typename T>
void PCCEncoder::CreateCoarseLayer( PCCImage<T, 3>&        image,
                                    PCCImage<T, 3>&        mip,
                                    std::vector<uint32_t>& occupancyMap,
                                    std::vector<uint32_t>& mipOccupancyMap ) {
  int dyadicWidth = 1;
  while ( dyadicWidth < image.getWidth() ) { dyadicWidth *= 2; }
  int dyadicHeight = 1;
  while ( dyadicHeight < image.getHeight() ) { dyadicHeight *= 2; }
  // allocate the mipmap with half the resolution
  mip.resize( ( dyadicWidth / 2 ), ( dyadicHeight / 2 ), PCCCOLORFORMAT::YUV444 );
  mipOccupancyMap.resize( ( dyadicWidth / 2 ) * ( dyadicHeight / 2 ), 0 );
  int    stride    = image.getWidth();
  int    newStride = ( dyadicWidth / 2 );
  int    x;
  int    y;
  int    i;
  int    j;
  double num[3];
  double den;
  for ( y = 0; y < mip.getHeight(); y++ ) {
    for ( x = 0; x < mip.getWidth(); x++ ) {
      num[0] = 0;
      num[1] = 0;
      num[2] = 0;
      den    = 0;
      for ( i = 0; i < 2; i++ ) {
        for ( j = 0; j < 2; j++ ) {
          int row = ( 2 * y + i ) < 0 ? 0 : ( 2 * y + i ) >= image.getHeight() ? image.getHeight() - 1 : ( 2 * y + i );
          int column = ( 2 * x + j ) < 0 ? 0 : ( 2 * x + j ) >= image.getWidth() ? image.getWidth() - 1 : ( 2 * x + j );
          if ( occupancyMap[column + stride * row] == 1 ) {
            den++;
            for ( int cc = 0; cc < 3; cc++ ) { num[cc] += image.getValue( cc, column, row ); }
          }
        }
      }
      if ( den > 0 ) {
        mipOccupancyMap[x + newStride * y] = 1;
        for ( int cc = 0; cc < 3; cc++ ) { mip.setValue( cc, x, y, std::round( num[cc] / den ) ); }
      }
    }
  }
}

template <typename T>
void PCCEncoder::regionFill( PCCImage<T, 3>& image, std::vector<uint32_t>& occupancyMap, PCCImage<T, 3>& imageLowRes ) {
  int                   stride        = image.getWidth();
  int                   numElem       = 0;
  int                   numSparseElem = 0;
  std::vector<uint32_t> indexing;
  indexing.resize( occupancyMap.size() );
  for ( int i = 0; i < occupancyMap.size(); i++ ) {
    if ( occupancyMap[i] == 0 ) {
      indexing[i] = numElem;
      numElem++;
    }
  }
  // create a sparse matrix with the coefficients
  std::vector<uint32_t> iSparse;
  std::vector<uint32_t> jSparse;
  std::vector<double>   valSparse;
  iSparse.resize( numElem * 5 );
  jSparse.resize( numElem * 5 );
  valSparse.resize( numElem * 5 );
  // create an initial solution using the low-resolution
  std::vector<double> b[3];
  b[0].resize( numElem );
  b[1].resize( numElem );
  b[2].resize( numElem );
  // fill in the system
  int idx       = 0;
  int idxSparse = 0;
  for ( int row = 0; row < image.getHeight(); row++ ) {
    for ( int column = 0; column < image.getWidth(); column++ ) {
      if ( occupancyMap[column + stride * row] == 0 ) {
        int count = 0;
        b[0][idx] = 0;
        b[1][idx] = 0;
        b[2][idx] = 0;
        for ( int i = -1; i < 2; i++ ) {
          for ( int j = -1; j < 2; j++ ) {
            if ( ( i == j ) || ( i == -j ) ) { continue; }
            if ( ( column + j < 0 ) || ( column + j > image.getWidth() - 1 ) ) { continue; }
            if ( ( row + i < 0 ) || ( row + i > image.getHeight() - 1 ) ) { continue; }
            count++;
            if ( occupancyMap[column + j + stride * ( row + i )] == 1 ) {
              b[0][idx] += image.getValue( 0, column + j, row + i );
              b[1][idx] += image.getValue( 1, column + j, row + i );
              b[2][idx] += image.getValue( 2, column + j, row + i );
            } else {
              iSparse[idxSparse]   = idx;
              jSparse[idxSparse]   = indexing[column + j + stride * ( row + i )];
              valSparse[idxSparse] = -1;
              idxSparse++;
            }
          }
        }
        // now insert the weight of the center pixel
        iSparse[idxSparse]   = idx;
        jSparse[idxSparse]   = idx;
        valSparse[idxSparse] = count;
        idx++;
        idxSparse++;
      }
    }
  }
  numSparseElem = idxSparse;
  // now solve the linear system Ax=b using Gauss-Siedel relaxation, with
  // initial guess coming from the lower
  // resolution
  std::vector<double> x[3];
  x[0].resize( numElem );
  x[1].resize( numElem );
  x[2].resize( numElem );
  if ( imageLowRes.getWidth() == image.getWidth() ) {
    // low resolution image not provided, let's use for the initialization the
    // mean value of the active pixels
    double mean[3] = {0.0, 0.0, 0.0};
    idx            = 0;
    for ( int row = 0; row < image.getHeight(); row++ ) {
      for ( int column = 0; column < image.getWidth(); column++ ) {
        if ( occupancyMap[column + stride * row] == 1 ) {
          mean[0] += double( image.getValue( 0, column, row ) );
          mean[1] += double( image.getValue( 1, column, row ) );
          mean[2] += double( image.getValue( 2, column, row ) );
          idx++;
        }
      }
    }
    mean[0] /= idx;
    mean[1] /= idx;
    mean[2] /= idx;
    idx = 0;
    for ( int row = 0; row < image.getHeight(); row++ ) {
      for ( int column = 0; column < image.getWidth(); column++ ) {
        if ( occupancyMap[column + stride * row] == 0 ) {
          x[0][idx] = mean[0];
          x[1][idx] = mean[1];
          x[2][idx] = mean[2];
          idx++;
        }
      }
    }
  } else {
    idx = 0;
    for ( int row = 0; row < image.getHeight(); row++ ) {
      for ( int column = 0; column < image.getWidth(); column++ ) {
        if ( occupancyMap[column + stride * row] == 0 ) {
          x[0][idx] = imageLowRes.getValue( 0, column / 2, row / 2 );
          x[1][idx] = imageLowRes.getValue( 1, column / 2, row / 2 );
          x[2][idx] = imageLowRes.getValue( 2, column / 2, row / 2 );
          idx++;
        }
      }
    }
  }
  int    maxIteration = 1024;
  double maxError     = 0.00001;
  for ( int cc = 0; cc < 3; cc++ ) {
    int it = 0;
    for ( ; it < maxIteration; it++ ) {
      int    idxSparse = 0;
      double error     = 0;
      double val       = 0;
      for ( int centerIdx = 0; centerIdx < numElem; centerIdx++ ) {
        // add the b result
        val = b[cc][centerIdx];
        while ( ( idxSparse < numSparseElem ) && ( iSparse[idxSparse] == centerIdx ) ) {
          if ( valSparse[idxSparse] < 0 ) {
            val += x[cc][jSparse[idxSparse]];
            idxSparse++;
          } else {
            // final value
            val /= valSparse[idxSparse];
            // accumulate the error
            error += ( val - x[cc][centerIdx] ) * ( val - x[cc][centerIdx] );
            // update the value
            x[cc][centerIdx] = val;
            idxSparse++;
          }
        }
      }
      error = error / numElem;
      if ( error < maxError ) { break; }
    }
  }
  // put the value back in the image
  idx = 0;
  for ( int row = 0; row < image.getHeight(); row++ ) {
    for ( int column = 0; column < image.getWidth(); column++ ) {
      if ( occupancyMap[column + stride * row] == 0 ) {
        image.setValue( 0, column, row, x[0][idx] );
        image.setValue( 1, column, row, x[1][idx] );
        image.setValue( 2, column, row, x[2][idx] );
        idx++;
      }
    }
  }
}

/* pull push filling algorithm */
template <typename T>
int PCCEncoder::mean4w( T             p1,
                        unsigned char w1,
                        T             p2,
                        unsigned char w2,
                        T             p3,
                        unsigned char w3,
                        T             p4,
                        unsigned char w4 ) {
  int result = ( p1 * int( w1 ) + p2 * int( w2 ) + p3 * int( w3 ) + p4 * int( w4 ) ) /
               ( int( w1 ) + int( w2 ) + int( w3 ) + int( w4 ) );
  return result;
}

// Generates a weighted mipmap
template <typename T>
void PCCEncoder::pushPullMip( const PCCImage<T, 3>&        image,
                              PCCImage<T, 3>&              mip,
                              const std::vector<uint32_t>& occupancyMap,
                              std::vector<uint32_t>&       mipOccupancyMap ) {
  unsigned char w1;
  unsigned char w2;
  unsigned char w3;
  unsigned char w4;
  unsigned char val1;
  unsigned char val2;
  unsigned char val3;
  unsigned char val4;
  const size_t  width     = image.getWidth();
  const size_t  height    = image.getHeight();
  const size_t  newWidth  = ( ( width + 1 ) >> 1 );
  const size_t  newHeight = ( ( height + 1 ) >> 1 );
  // allocate the mipmap with half the resolution
  mip.resize( newWidth, newHeight, PCCCOLORFORMAT::YUV444 );
  mipOccupancyMap.resize( newWidth * newHeight, 0 );
  for ( size_t y = 0; y < newHeight; ++y ) {
    const size_t yUp = y << 1;
    for ( size_t x = 0; x < newWidth; ++x ) {
      const size_t xUp = x << 1;
      if ( occupancyMap[xUp + width * yUp] == 0 ) {
        w1 = 0;
      } else {
        w1 = 255;
      }
      if ( ( xUp + 1 >= width ) || ( occupancyMap[xUp + 1 + width * yUp] == 0 ) ) {
        w2 = 0;
      } else {
        w2 = 255;
      }
      if ( ( yUp + 1 >= height ) || ( occupancyMap[xUp + width * ( yUp + 1 )] == 0 ) ) {
        w3 = 0;
      } else {
        w3 = 255;
      }
      if ( ( xUp + 1 >= width ) || ( yUp + 1 >= height ) || ( occupancyMap[xUp + 1 + width * ( yUp + 1 )] == 0 ) ) {
        w4 = 0;
      } else {
        w4 = 255;
      }
      if ( w1 + w2 + w3 + w4 > 0 ) {
        for ( int cc = 0; cc < 3; cc++ ) {
          val1 = image.getValue( cc, xUp, yUp );
          if ( xUp + 1 >= width ) {
            val2 = 0;
          } else {
            val2 = image.getValue( cc, xUp + 1, yUp );
          }
          if ( yUp + 1 >= height ) {
            val3 = 0;
          } else {
            val3 = image.getValue( cc, xUp, yUp + 1 );
          }
          if ( ( xUp + 1 >= width ) || ( yUp + 1 >= height ) ) {
            val4 = 0;
          } else {
            val4 = image.getValue( cc, xUp + 1, yUp + 1 );
          }
          T newVal = mean4w( val1, w1, val2, w2, val3, w3, val4, w4 );
          mip.setValue( cc, x, y, newVal );
        }
        mipOccupancyMap[x + newWidth * y] = 1;
      }
    }
  }
}

// interpolate using mipmap
template <typename T>
void PCCEncoder::pushPullFill( PCCImage<T, 3>&              image,
                               const PCCImage<T, 3>&        mip,
                               const std::vector<uint32_t>& occupancyMap,
                               int                          numIters ) {
  const size_t width    = mip.getWidth();
  const size_t height   = mip.getHeight();
  const size_t widthUp  = image.getWidth();
  const size_t heightUp = image.getHeight();
  assert( ( ( widthUp + 1 ) >> 1 ) == width );
  assert( ( ( heightUp + 1 ) >> 1 ) == height );
  int           x;
  int           y;
  int           xUp;
  int           yUp;
  unsigned char w1;
  unsigned char w2;
  unsigned char w3;
  unsigned char w4;
  for ( yUp = 0; yUp < heightUp; ++yUp ) {
    y = yUp >> 1;
    for ( xUp = 0; xUp < widthUp; ++xUp ) {
      x = xUp >> 1;
      if ( occupancyMap[xUp + widthUp * yUp] == 0 ) {
        if ( ( xUp % 2 == 0 ) && ( yUp % 2 == 0 ) ) {
          w1 = 144;
          w2 = ( x > 0 ? static_cast<unsigned char>( 48 ) : 0 );
          w3 = ( y > 0 ? static_cast<unsigned char>( 48 ) : 0 );
          w4 = ( ( ( x > 0 ) && ( y > 0 ) ) ? static_cast<unsigned char>( 16 ) : 0 );
          for ( int cc = 0; cc < 3; cc++ ) {
            T val       = mip.getValue( cc, x, y );
            T valLeft   = ( x > 0 ? mip.getValue( cc, x - 1, y ) : 0 );
            T valUp     = ( y > 0 ? mip.getValue( cc, x, y - 1 ) : 0 );
            T valUpLeft = ( ( x > 0 && y > 0 ) ? mip.getValue( cc, x - 1, y - 1 ) : 0 );
            T newVal    = mean4w( val, w1, valLeft, w2, valUp, w3, valUpLeft, w4 );
            image.setValue( cc, xUp, yUp, newVal );
          }
        } else if ( ( xUp % 2 == 1 ) && ( yUp % 2 == 0 ) ) {
          w1 = 144;
          w2 = ( x < width - 1 ? static_cast<unsigned char>( 48 ) : 0 );
          w3 = ( y > 0 ? static_cast<unsigned char>( 48 ) : 0 );
          w4 = ( ( ( x < width - 1 ) && ( y > 0 ) ) ? static_cast<unsigned char>( 16 ) : 0 );
          for ( int cc = 0; cc < 3; cc++ ) {
            T val        = mip.getValue( cc, x, y );
            T valRight   = ( x < width - 1 ? mip.getValue( cc, x + 1, y ) : 0 );
            T valUp      = ( y > 0 ? mip.getValue( cc, x, y - 1 ) : 0 );
            T valUpRight = ( ( ( x < width - 1 ) && ( y > 0 ) ) ? mip.getValue( cc, x + 1, y - 1 ) : 0 );
            T newVal     = mean4w( val, w1, valRight, w2, valUp, w3, valUpRight, w4 );
            image.setValue( cc, xUp, yUp, newVal );
          }
        } else if ( ( xUp % 2 == 0 ) && ( yUp % 2 == 1 ) ) {
          w1 = 144;
          w2 = ( x > 0 ? static_cast<unsigned char>( 48 ) : 0 );
          w3 = ( y < height - 1 ? static_cast<unsigned char>( 48 ) : 0 );
          w4 = ( ( ( x > 0 ) && ( y < height - 1 ) ) ? static_cast<unsigned char>( 16 ) : 0 );
          for ( int cc = 0; cc < 3; cc++ ) {
            T val         = mip.getValue( cc, x, y );
            T valLeft     = ( x > 0 ? mip.getValue( cc, x - 1, y ) : 0 );
            T valDown     = ( ( y < height - 1 ) ? mip.getValue( cc, x, y + 1 ) : 0 );
            T valDownLeft = ( ( x > 0 && ( y < height - 1 ) ) ? mip.getValue( cc, x - 1, y + 1 ) : 0 );
            T newVal      = mean4w( val, w1, valLeft, w2, valDown, w3, valDownLeft, w4 );
            image.setValue( cc, xUp, yUp, newVal );
          }
        } else {
          w1 = 144;
          w2 = ( x < width - 1 ? static_cast<unsigned char>( 48 ) : 0 );
          w3 = ( y < height - 1 ? static_cast<unsigned char>( 48 ) : 0 );
          w4 = ( ( ( x < width - 1 ) && ( y < height - 1 ) ) ? static_cast<unsigned char>( 16 ) : 0 );
          for ( int cc = 0; cc < 3; cc++ ) {
            T val          = mip.getValue( cc, x, y );
            T valRight     = ( x < width - 1 ? mip.getValue( cc, x + 1, y ) : 0 );
            T valDown      = ( ( y < height - 1 ) ? mip.getValue( cc, x, y + 1 ) : 0 );
            T valDownRight = ( ( ( x < width - 1 ) && ( y < height - 1 ) ) ? mip.getValue( cc, x + 1, y + 1 ) : 0 );
            T newVal       = mean4w( val, w1, valRight, w2, valDown, w3, valDownRight, w4 );
            image.setValue( cc, xUp, yUp, newVal );
          }
        }
      }
    }
  }
  auto tmpImage( image );
  for ( size_t n = 0; n < numIters; n++ ) {
    for ( int y = 0; y < heightUp; y++ ) {
      for ( int x = 0; x < widthUp; x++ ) {
        if ( occupancyMap[x + widthUp * y] == 0 ) {
          int x1 = ( x > 0 ) ? x - 1 : x;
          int y1 = ( y > 0 ) ? y - 1 : y;
          int x2 = ( x < widthUp - 1 ) ? x + 1 : x;
          int y2 = ( y < heightUp - 1 ) ? y + 1 : y;
          for ( size_t c = 0; c < 3; c++ ) {
            int val = image.getValue( c, x1, y1 ) + image.getValue( c, x2, y1 ) + image.getValue( c, x1, y2 ) +
                      image.getValue( c, x2, y2 ) + image.getValue( c, x1, y ) + image.getValue( c, x2, y ) +
                      image.getValue( c, x, y1 ) + image.getValue( c, x, y2 );
            tmpImage.setValue( c, x, y, ( val + 4 ) >> 3 );
          }
        }
      }
    }
    swap( image, tmpImage );
  }
}

template <typename T>
void PCCEncoder::dilateSmoothedPushPull( PCCFrameContext& frame, PCCImage<T, 3>& image, int mapIdx ) {
  auto                               occupancyMapTemp = frame.getOccupancyMap();
  int                                i                = 0;
  std::vector<PCCImage<T, 3>>        mipVec;
  std::vector<std::vector<uint32_t>> mipOccupancyMapVec;
  int                                div    = 2;
  int                                miplev = 0;

  // pull phase create the mipmap
  while ( true ) {
    mipVec.resize( mipVec.size() + 1 );
    mipOccupancyMapVec.resize( mipOccupancyMapVec.size() + 1 );
    div *= 2;
    if ( miplev > 0 ) {
      pushPullMip( mipVec[miplev - 1], mipVec[miplev], mipOccupancyMapVec[miplev - 1], mipOccupancyMapVec[miplev] );
    } else {
      pushPullMip( image, mipVec[miplev], occupancyMapTemp, mipOccupancyMapVec[miplev] );
    }
    if ( mipVec[miplev].getWidth() <= 4 || mipVec[miplev].getHeight() <= 4 ) { break; }
    ++miplev;
  }
  miplev++;
#if DEBUG_PATCH
  for ( int k = 0; k < miplev; k++ ) {
    char buf[100];
    sprintf( buf, "mip%02i", k );
    std::string filename = addVideoFormat( buf, mipVec[k].getWidth(), mipVec[k].getHeight(), false, false );
    mipVec[k].write( filename, 1 );
  }
#endif
  // push phase: refill
  int numIters = 4;
  for ( i = miplev - 1; i >= 0; --i ) {
    if ( i > 0 ) {
      pushPullFill( mipVec[i - 1], mipVec[i], mipOccupancyMapVec[i - 1], numIters );
    } else {
      pushPullFill( image, mipVec[i], occupancyMapTemp, numIters );
    }
    numIters = ( std::min )( numIters + 1, 16 );
  }
#if DEBUG_PATCH
  for ( int k = 0; k < miplev; k++ ) {
    char buf[100];
    sprintf( buf, "mipfill%02i", k );
    std::string filename = addVideoFormat( buf, mipVec[k].getWidth(), mipVec[k].getHeight(), false, false );
    mipVec[k].write( filename, 1 );
  }
#endif
}

void PCCEncoder::presmoothPointCloudColor( PCCPointSet3& reconstruct, const PCCEncoderParameters params ) {
  const size_t            pointCount = reconstruct.getPointCount();
  PCCKdTree               kdtree( reconstruct );
  PCCNNResult             result;
  std::vector<PCCColor3B> temp;
  temp.resize( pointCount );
  for ( size_t m = 0; m < pointCount; ++m ) { temp[m] = reconstruct.getColor( m ); }
  tbb::task_arena limited( static_cast<int>( params.nbThread_ ) );
  limited.execute( [&] {
    tbb::parallel_for( size_t( 0 ), pointCount, [&]( const size_t i ) {
      //  for (size_t i = 0; i < pointCount; ++i) {
      PCCNNResult result;
      if ( reconstruct.getBoundaryPointType( i ) == 2 ) {
        kdtree.searchRadius( reconstruct[i], params.neighborCountColorPreSmoothing_, params.radius2ColorPreSmoothing_,
                             result );
        PCCVector3D          centroid( 0.0 );
        size_t               neighborCount = 0;
        std::vector<uint8_t> Lum;
        for ( size_t r = 0; r < result.count(); ++r ) {
          const double dist2 = result.dist( r );
          if ( dist2 > params.radius2ColorPreSmoothing_ ) { break; }
          ++neighborCount;
          const size_t index = result.indices( r );
          PCCColor3B   color = reconstruct.getColor( index );
          centroid[0] += double( color[0] );
          centroid[1] += double( color[1] );
          centroid[2] += double( color[2] );
          double Y = 0.2126 * double( color[0] ) + 0.7152 * double( color[1] ) + 0.0722 * double( color[2] );
          Lum.push_back( uint8_t( Y ) );
        }

        PCCColor3B color;
        if ( neighborCount != 0u ) {
          for ( size_t k = 0; k < 3; ++k ) {
            centroid[k] = double( int64_t( centroid[k] + ( neighborCount / 2 ) ) / neighborCount );
          }

          // Texture characterization
          double     H               = entropy( Lum, int( neighborCount ) );
          PCCColor3B colorQP         = reconstruct.getColor( i );
          double     distToCentroid2 = 0;
          for ( size_t k = 0; k < 3; ++k ) { distToCentroid2 += abs( centroid[k] - double( colorQP[k] ) ); }
          if ( distToCentroid2 >= double( params.thresholdColorPreSmoothing_ ) &&
               H < double( params.thresholdColorPreSmoothingLocalEntropy_ ) ) {
            color[0] = uint8_t( centroid[0] );
            color[1] = uint8_t( centroid[1] );
            color[2] = uint8_t( centroid[2] );
            temp[i]  = color;
          }
        }
      }
    } );
  } );

  limited.execute( [&] {
    tbb::parallel_for( size_t( 0 ), pointCount, [&]( const size_t i ) {
      // for (size_t i = 0; i < pointCount; ++i) {
      reconstruct.setColor( i, temp[i] );
    } );
  } );
}

bool PCCEncoder::generateTextureVideo( const PCCGroupOfFrames&     sources,
                                       PCCGroupOfFrames&           reconstructs,
                                       PCCContext&                 context,
                                       const PCCEncoderParameters& params ) {
  auto& video   = context.getVideoTextureMultiple()[0];
  auto& videoT1 = context.getVideoTextureMultiple()[1];
  if ( params_.multipleStreams_ ) {
    video.resize( context.size() );
    videoT1.resize( context.size() );
  } else {
    video.resize( context.size() * ( params.mapCountMinus1_ + 1 ) );
  }
  bool ret = true;
  for ( size_t i = 0; i < context.size(); i++ ) {
    auto&  frame    = context[i].getTitleFrameContext();
    size_t mapCount = params_.mapCountMinus1_ + 1;
    sources[i].transferColors( reconstructs[i], int32_t( params_.bestColorSearchRange_ ),
                               static_cast<int>( params_.losslessGeo_ ) == 1, params_.numNeighborsColorTransferFwd_,
                               params_.numNeighborsColorTransferBwd_, params_.useDistWeightedAverageFwd_,
                               params_.useDistWeightedAverageBwd_, params_.skipAvgIfIdenticalSourcePointPresentFwd_,
                               params_.skipAvgIfIdenticalSourcePointPresentBwd_, params_.distOffsetFwd_,
                               params_.distOffsetBwd_, params_.maxGeometryDist2Fwd_, params_.maxGeometryDist2Bwd_,
                               params_.maxColorDist2Fwd_, params_.maxColorDist2Bwd_, params_.excludeColorOutlier_,
                               params_.thresholdColorOutlierDist_ );
    // color pre-smoothing
    if ( !params_.losslessGeo_ && params_.flagColorPreSmoothing_ ) {
      presmoothPointCloudColor( reconstructs[i], params );
    }
    size_t imageWidth  = frame.getWidth();
    size_t imageHeight = frame.getHeight();
    if ( params_.multipleStreams_ ) {
      auto& image = video.getFrame( i );
      image.resize( imageWidth, imageHeight, PCCCOLORFORMAT::RGB444 );
      image.set( 0 );
      auto& image1 = videoT1.getFrame( i );
      image1.resize( imageWidth, imageHeight, PCCCOLORFORMAT::RGB444 );
      image1.set( 0 );
    } else {
      for ( size_t f = 0; f < mapCount; ++f ) {
        auto& image = video.getFrame( f + mapCount * i );
        image.resize( imageWidth, imageHeight, PCCCOLORFORMAT::RGB444 );
        image.set( 0 );
      }
    }
    size_t accTilePointCount = 0;
    for ( size_t tileIdx = 0; tileIdx < context[i].getNumTilesInAtlasFrame(); tileIdx++ ) {
      accTilePointCount =
          generateTextureVideo( reconstructs[i], context, i, tileIdx, video, videoT1, mapCount, accTilePointCount );
    }
  }
  return ret;
}

size_t PCCEncoder::generateTextureVideo( const PCCPointSet3& reconstruct,
                                         PCCContext&         context,
                                         size_t              frameIndex,
                                         size_t              tileIndex,
                                         PCCVideoTexture&    video,
                                         PCCVideoTexture&    videoT1,
                                         const size_t        mapCount,
                                         size_t              accTilePointCount ) {
  auto&  tile          = context[frameIndex].getTile( tileIndex );
  auto&  pointToPixel  = context[frameIndex].getTitleFrameContext().getPointToPixel();
  size_t regPointCount = tile.getTotalNumberOfRegularPoints();
  size_t auxPointCount = tile.getTotalNumberOfEOMPoints() + tile.getTotalNumberOfRawPoints();
  size_t pointCount    = regPointCount + ( params_.useRawPointsSeparateVideo_ ? 0 : auxPointCount );

  bool lossyRawPointsPatch = tile.getRawPatchEnabledFlag() && ( !params_.losslessGeo_ );
  if ( ( pointCount == 0u && auxPointCount == 0u ) || !reconstruct.hasColors() ) { return false; }
  std::vector<bool> markT1;
  if ( params_.mapCountMinus1_ > 0 && params_.removeDuplicatePoints_ ) {
    const size_t size = tile.getWidth() * tile.getHeight();
    markT1.resize( size );
    for ( size_t i = 0; i < size; i++ ) { markT1[i] = false; }
  }
  for ( size_t i = 0; i < pointCount; ++i ) {
    const PCCVector3<size_t> location = pointToPixel[accTilePointCount + i];
    const PCCColor3B         color    = reconstruct.getColor( accTilePointCount + i );
    const size_t             u        = tile.getLeftTopXInFrame() + location[0];
    const size_t             v        = tile.getLeftTopYInFrame() + location[1];
    const size_t             f        = location[2];
    if ( params_.singleMapPixelInterleaving_ ) {
      if ( ( f == 0 && ( ( u + v ) % 2 == 0 ) ) || ( f == 1 && ( ( u + v ) % 2 == 1 ) ) ) {
        auto& image = video.getFrame( frameIndex );
        image.setValue( 0, u, v, color[0] );
        image.setValue( 1, u, v, color[1] );
        image.setValue( 2, u, v, color[2] );
      }
    } else {
      if ( f < mapCount ) {
        auto& image = params_.multipleStreams_
                          ? ( ( f == 0 ) ? video.getFrame( frameIndex ) : videoT1.getFrame( frameIndex ) )
                          : video.getFrame( f + frameIndex * mapCount );
        image.setValue( 0, u, v, color[0] );
        image.setValue( 1, u, v, color[1] );
        image.setValue( 2, u, v, color[2] );

        if ( params_.mapCountMinus1_ > 0 && params_.removeDuplicatePoints_ ) {
          auto& image1 =
              params_.multipleStreams_ ? videoT1.getFrame( frameIndex ) : video.getFrame( 1 + frameIndex * mapCount );
          if ( f == 0 ) {
            if ( !markT1[location[1] * tile.getWidth() + location[0]] ) {
              image1.setValue( 0, u, v, color[0] );
              image1.setValue( 1, u, v, color[1] );
              image1.setValue( 2, u, v, color[2] );
            }
          } else {
            markT1[location[1] * tile.getWidth() + location[0]] = true;
          }
        }
      }
    }
  }

  if ( ( params_.losslessGeo_ || lossyRawPointsPatch ) && params_.useRawPointsSeparateVideo_ ) {
    size_t numOfRegularPoints = tile.getTotalNumberOfRegularPoints();
    size_t numberOfEOMPoints  = tile.getTotalNumberOfEOMPoints();
    size_t numOfRawGeos       = tile.getTotalNumberOfRawPoints();
    // raw points
    std::cout.flush();
    std::vector<PCCColor3B>& rawTextures = tile.getRawPointsTextures();
    std::vector<PCCColor3B>& eomTextures = tile.getEOMTextures();
    std::cout.flush();
    rawTextures.resize( numOfRawGeos );
    eomTextures.resize( numberOfEOMPoints );
    for ( size_t i = 0; i < numOfRawGeos; ++i ) {
      const PCCColor3B color = reconstruct.getColor( accTilePointCount + numOfRegularPoints + numberOfEOMPoints + i );
      rawTextures[i]         = color;
    }
    if ( params_.enhancedOccupancyMapCode_ ) {
      for ( size_t i = 0; i < numberOfEOMPoints; ++i ) {
        const PCCColor3B color = reconstruct.getColor( accTilePointCount + numOfRegularPoints + i );
        eomTextures[i]         = color;
      }
    }
  }
  return accTilePointCount + regPointCount + auxPointCount;
}

void PCCEncoder::performDataAdaptiveGPAMethod( PCCContext& context,
                                               size_t      tileIndex,
                                               size_t      frameWidthIn,
                                               size_t      frameHeightIn,
                                               int         firstFrame,
                                               int         lastFramePlus1 ) {
  // some valid parameters;
  SubContext    subContextPre;
  SubContext    subContextCur;  // [start, end);
  unionPatch    unionPatchPre;
  unionPatch    unionPatchCur;                 // [trackIndex, patchUnion];
  GlobalPatches globalPatchTracks;             // [trackIndex, <[frameIndex, patchIndex]>];
  bool          startSubContext      = true;   // startSubContext is initialized as true;  start a subContext;
  bool          endSubContext        = false;  // endSubContext   is initialized as false; end   a subContext;
  int           preSubcontextFrameId = -1;

  // iterate over all frameContexts;
  for ( size_t frameIndex = ( firstFrame >= 0 ? firstFrame : 0 );
        frameIndex < ( lastFramePlus1 >= 0 ? lastFramePlus1 : context.size() ); ++frameIndex ) {
    std::cout << "\tGPA for [" << firstFrame << "~" << lastFramePlus1 << ") Tile [" << tileIndex << "] Frame["
              << frameIndex << "] initial size:" << context.getFrame( frameIndex ).getTile( tileIndex ).getWidth()
              << "x" << context.getFrame( frameIndex ).getTile( tileIndex ).getHeight() << std::endl;
    auto& tile = context.getFrame( frameIndex ).getTile( tileIndex );

    bool useRefFrame = true /*params_.keepGPARotation_*/;
    // determine whether start a subContext or not;
    if ( startSubContext ) {
      initializeSubContext( tile, subContextPre, globalPatchTracks, unionPatchPre, frameIndex );

      if ( subContextPre.first == 0 ) { useRefFrame = false; }
      packingFirstFrame( context, frameIndex, tileIndex, tile.getWidth(), tile.getHeight(), params_.packingStrategy_,
                         params_.safeGuardDistance_, useRefFrame );

      context[subContextPre.first].getTile( tileIndex ).getPrePCCGPAFrameSize() =
          context[subContextPre.first].getTile( tileIndex ).getCurPCCGPAFrameSize();
      context[subContextPre.first].getTile( tileIndex ).getCurPCCGPAFrameSize().widthGPA_  = 0;
      context[subContextPre.first].getTile( tileIndex ).getCurPCCGPAFrameSize().heightGPA_ = 0;
      for ( auto& patch : context[subContextPre.first].getTile( tileIndex ).getPatches() ) {
        patch.getPreGPAPatchData() = patch.getCurGPAPatchData();
        patch.getCurGPAPatchData().initialize();
      }
      if ( frameIndex == ( lastFramePlus1 >= 0 ? ( lastFramePlus1 - 1 ) : ( context.size() - 1 ) ) ) {
        context.getSubContexts().emplace_back( subContextPre );
        updatePatchInformation( context, tileIndex, subContextPre );
        break;
      }
      subContextCur   = subContextPre;
      startSubContext = false;
      continue;
    }

    subContextCur.first  = subContextPre.first;
    subContextCur.second = frameIndex + 1;
    preSubcontextFrameId = subContextCur.first - 1;
    if ( subContextCur.first == 0 ) {
      useRefFrame          = false;
      preSubcontextFrameId = -1;
    }

    // clear current information;
    clearCurrentGPAPatchDataInfor( context, tileIndex, subContextCur );

    // genrate globalPatchTracks;
    size_t preIndex =
        frameIndex - subContextCur.first - 1;  // preIndex is the previous index in the current subcontext.
    generateGlobalPatches( context, frameIndex, tileIndex, globalPatchTracks, preIndex );

    // patch unions generation and packing;
    size_t unionsHeight =
        unionPatchGenerationAndPacking( globalPatchTracks, context, tileIndex, tile.getWidth(), tile.getHeight(),
                                        unionPatchCur, preSubcontextFrameId, params_.safeGuardDistance_, useRefFrame );

    // perform GPA packing;
    bool badPatchCount   = false;
    bool badUnionsHeight = false;
    bool badGPAPacking   = false;
    if ( double( unionPatchCur.size() ) / globalPatchTracks.size() < 0.15 ) { badPatchCount = true; }
    if ( unionsHeight > ( context[frameIndex].getNumTilesInAtlasFrame() == 1 ? frameHeightIn : tile.getHeight() ) ) {
      badUnionsHeight = true;
    }
    if ( printDetailedInfo ) {
      std::cout << "badPatchCount: " << badPatchCount << "badUnionsHeight: " << badUnionsHeight << std::endl;
    }
    if ( !badPatchCount && !badUnionsHeight ) {
      // patch information updating;
      updateGPAPatchInformation( context, tileIndex, subContextCur, unionPatchCur );
      // save the data into preGPAPatchData.
      if ( unionsHeight > ( context[frameIndex].getNumTilesInAtlasFrame() == 1 ? frameHeightIn : tile.getHeight() ) ) {
        badUnionsHeight = true;
      }
      performGPAPacking( subContextCur, unionPatchCur, context, tileIndex, frameWidthIn, frameHeightIn, badGPAPacking,
                         unionsHeight, params_.safeGuardDistance_, useRefFrame );
    }

    endSubContext = ( badPatchCount || badUnionsHeight || badGPAPacking );
    std::cout << "\tThe endSubContext is: " << endSubContext << "=(" << badPatchCount << "||" << badUnionsHeight << "||"
              << badGPAPacking << ")" << std::endl;
    if ( endSubContext ) {
      std::cout << "\tThe frame is a end point --- " << frameIndex << std::endl;
      // clear current information;
      clearCurrentGPAPatchDataInfor( context, tileIndex, subContextCur );
      assert( subContextCur.second - subContextCur.first > 1 );

      subContextCur.first  = 0;
      subContextCur.second = 0;
      unionPatchCur.clear();
      globalPatchTracks.clear();  // GlobalPatches.......;
      // retain previous information;
      context.getSubContexts().emplace_back( subContextPre );  // SubContext..........;
      startSubContext = true;
      endSubContext   = false;
      frameIndex -= 1;  // should stay at the start point for next subcontext.

      // update Patch information;
      updatePatchInformation( context, tileIndex, subContextPre );
    } else {
      std::cout << "\tThe frame " << frameIndex << " is not a end point ---" << std::endl;
      // previous information updating;
      for ( size_t j = subContextCur.first; j < subContextCur.second; ++j ) {
        auto& tile                   = context[j].getTile( tileIndex );
        auto& curPatches             = tile.getPatches();
        tile.getPrePCCGPAFrameSize() = tile.getCurPCCGPAFrameSize();
        assert( !curPatches.empty() );
        for ( auto& curPatch : curPatches ) { curPatch.getPreGPAPatchData() = curPatch.getCurGPAPatchData(); }
        if ( !tile.getRawPointsPatches().empty() && !tile.getUseRawPointsSeparateVideo() ) {
          for ( size_t idxRawPatches = 0; idxRawPatches < tile.getRawPointsPatches().size(); idxRawPatches++ ) {
            tile.getRawPointsPatch( idxRawPatches ).preV0_ = tile.getRawPointsPatch( idxRawPatches ).tempV0_;
          }
        }
      }
      subContextPre = subContextCur;
      unionPatchPre.clear();
      unionPatchPre = unionPatchCur;
      std::cout << "\tcleared current tried infor:" << std::endl;
      // clear current information;
      for ( size_t j = subContextCur.first; j < subContextCur.second; ++j ) {
        auto& curPatches = context[j].getTile( tileIndex ).getPatches();
        assert( !curPatches.empty() );
        for ( auto& curPatch : curPatches ) { curPatch.getCurGPAPatchData().initialize(); }
      }
      subContextCur.first  = 0;
      subContextCur.second = 0;
      unionPatchCur.clear();
      // the ending......;
      if ( frameIndex == ( lastFramePlus1 < 0 ? ( context.size() - 1 ) : ( lastFramePlus1 - 1 ) ) ) {
        context.getSubContexts().emplace_back( subContextPre );  // SubContext..........;
        std::cout << "\tThis is the last frame......." << std::endl;

        // update information;
        updatePatchInformation( context, tileIndex, subContextPre );
        break;
      }
    }
  }
#if 1
  printf( "out of the GPA loop\n" );
#endif
}

void PCCEncoder::initializeSubContext( PCCFrameContext& tile,
                                       SubContext&      subContext,
                                       GlobalPatches&   globalPatchTracks,
                                       unionPatch&      unionPatch,
                                       size_t           frameIndex ) {
  // 1. initialize subContext;
  subContext.first  = frameIndex;
  subContext.second = frameIndex + 1;
  std::cout << "\tNew subContext:[" << subContext.first << "," << subContext.second << ")" << std::endl;

  // 2. initialize globalPatchTracks && unionPatch;
  unionPatch.clear();
  globalPatchTracks.clear();
  for ( size_t patchIndex = 0; patchIndex < tile.getPatches().size(); ++patchIndex ) {
    globalPatchTracks[patchIndex].emplace_back( std::make_pair( frameIndex, patchIndex ) );
    tile.getPatches()[patchIndex].getCurGPAPatchData().isGlobalPatch    = true;
    tile.getPatches()[patchIndex].getCurGPAPatchData().globalPatchIndex = patchIndex;
  }
}
void PCCEncoder::clearCurrentGPAPatchDataInfor( PCCContext& context, size_t tileIndex, SubContext& subContext ) {
  // clear current information;
  for ( size_t j = subContext.first; j < subContext.second; ++j ) {
    auto& curPatches = context[j].getTile( tileIndex ).getPatches();
    assert( !curPatches.empty() );
    for ( auto& curPatch : curPatches ) { curPatch.getCurGPAPatchData().initialize(); }
    if ( !context[j].getTile( tileIndex ).getRawPointsPatches().empty() ) {
      for ( auto rawPointsPatch : context[j].getTile( tileIndex ).getRawPointsPatches() ) {
        rawPointsPatch.tempV0_ = 0;
      }
    }
  }
}
void PCCEncoder::generateGlobalPatches( PCCContext&    context,
                                        size_t         frameIndex,
                                        size_t         tileIndex,
                                        GlobalPatches& globalPatchTracks,
                                        size_t         preIndex ) {
  auto& curPatches = context[frameIndex].getTile( tileIndex ).getPatches();
  assert( !curPatches.empty() );
  for ( auto& globalPatchTrack : globalPatchTracks ) {
    auto& trackPatches = globalPatchTrack.second;  // !!!< <frameIndex, patchIndex> >;
    if ( trackPatches.empty() ) { continue; }
    const auto& preGlobalPatch = trackPatches[preIndex];
    const auto& prePatch       = context[preGlobalPatch.first].getTile( tileIndex ).getPatches()[preGlobalPatch.second];
    float       thresholdIOU   = 0.2F;
    float       maxIou         = 0.0F;
    int32_t     bestIdx        = -1;       // best matched patch index in curPatches;
    int32_t     cId            = 0;        // patch index in curPatches;
    for ( auto& curPatch : curPatches ) {  // curPatches; may be modified;
      if ( prePatch.getViewId() == curPatch.getViewId() && !( curPatch.getCurGPAPatchData().isMatched ) &&
           ( prePatch.getLodScaleX() == curPatch.getLodScaleX() &&
             prePatch.getLodScaleY() == curPatch.getLodScaleY() ) ) {
        Rect  preRect = Rect( prePatch.getU1(), prePatch.getV1(), prePatch.getSizeU(), prePatch.getSizeV() );
        Rect  curRect = Rect( curPatch.getU1(), curPatch.getV1(), curPatch.getSizeU(), curPatch.getSizeV() );
        float iou     = computeIOU( preRect, curRect );
        if ( iou > maxIou ) {
          maxIou  = iou;
          bestIdx = cId;
        }
      }
      cId++;
    }
    if ( maxIou > thresholdIOU ) {                                // !!!best match found;
      curPatches[bestIdx].getCurGPAPatchData().isMatched = true;  // indicating the patch is already matched;
      trackPatches.emplace_back( std::make_pair( frameIndex, bestIdx ) );
    } else {
      trackPatches.clear();
    }
  }

  // update global patch information according to curGlobalPatches;
  for ( auto& globalPatchTrack : globalPatchTracks ) {
    const size_t trackIndex   = globalPatchTrack.first;
    const auto&  trackPatches = globalPatchTrack.second;  // !!!< <frameIndex, patchIndex> >;
    if ( trackPatches.empty() ) { continue; }
    for ( const auto& trackPatch : trackPatches ) {
      GPAPatchData& curGPAPatchData =
          context[trackPatch.first].getTile( tileIndex ).getPatches()[trackPatch.second].getCurGPAPatchData();
      curGPAPatchData.isGlobalPatch    = true;
      curGPAPatchData.globalPatchIndex = trackIndex;
    }
  }
}

size_t PCCEncoder::unionPatchGenerationAndPacking( const GlobalPatches& globalPatchTracks,
                                                   PCCContext&          context,
                                                   size_t               tileIndex,
                                                   size_t               frameWidth,
                                                   size_t               frameHeight,
                                                   unionPatch&          unionPatchTemp,
                                                   size_t               refFrameIdx,
                                                   int                  safeguard,
                                                   bool                 useRefFrame ) {
  // 1. unionPatch generation;
  unionPatchTemp.clear();
  // 1.1 patchTracks generation;
  std::map<size_t, std::vector<PCCPatch>> patchTracks;
  for ( const auto& globalPatchTrack : globalPatchTracks ) {
    const auto& trackIndex   = globalPatchTrack.first;
    const auto& trackPatches = globalPatchTrack.second;
    if ( trackPatches.empty() ) { continue; }
    for ( const auto& trackPatch : trackPatches ) {
      patchTracks[trackIndex].emplace_back(
          context[trackPatch.first].getTile( tileIndex ).getPatches()[trackPatch.second] );
    }
  }
  // 1.2 union processing --- patchTracks -> unionPatch;
  for ( const auto& patchTrack : patchTracks ) {
    const auto& trackIndex   = patchTrack.first;
    const auto& trackPatches = patchTrack.second;
    assert( !trackPatches.empty() );
    // get the sizeU0 && sizeV0;
    size_t maxSizeU0 = 0;
    size_t maxSizeV0 = 0;
    for ( const auto& trackPatch : trackPatches ) {
      maxSizeU0 = std::max<size_t>( maxSizeU0, trackPatch.getSizeU0() );
      maxSizeV0 = std::max<size_t>( maxSizeV0, trackPatch.getSizeV0() );
    }
    // get the patch union;
    PCCPatch curPatchUnion;
    curPatchUnion.getIndex()  = trackIndex;
    curPatchUnion.getSizeU0() = maxSizeU0;
    curPatchUnion.getSizeV0() = maxSizeV0;
    curPatchUnion.getOccupancy().resize( maxSizeU0 * maxSizeV0, false );
    if ( useRefFrame && ( !trackPatches.empty() ) ) {
      assert( refFrameIdx != -1 );
      size_t matchedPatchIdx = trackPatches[0].getBestMatchIdx();  // the first frame in the subcontext.
      if ( matchedPatchIdx == -1 ) {
        curPatchUnion.getPatchOrientation() = -1;
      } else {  // suppose the refFrame is the same frame for all patches.
        curPatchUnion.getPatchOrientation() =
            context[refFrameIdx].getTile( tileIndex ).getPatches()[matchedPatchIdx].getPatchOrientation();
        if ( printDetailedInfo ) {
          std::cout << "Maintained orientation for "
                       "curPatchUnion.getPatchOrientation() = "
                    << curPatchUnion.getPatchOrientation() << std::endl;
        }
      }
    }
    for ( const auto& trackPatch : trackPatches ) {
      const auto& occupancy = trackPatch.getOccupancy();
      for ( size_t v = 0; v < trackPatch.getSizeV0(); ++v ) {
        for ( size_t u = 0; u < trackPatch.getSizeU0(); ++u ) {
          assert( v < maxSizeV0 );
          assert( u < maxSizeU0 );
          size_t p  = v * trackPatch.getSizeU0() + u;
          size_t up = v * curPatchUnion.getSizeU0() + u;
          if ( occupancy[p] && !( curPatchUnion.getOccupancy()[up] ) ) { curPatchUnion.getOccupancy()[up] = true; }
        }
      }
    }
    unionPatchTemp[trackIndex] = curPatchUnion;
  }

  // 2. unionPatch packing;
  size_t occupancySizeU = frameWidth / params_.occupancyResolution_;
  size_t occupancySizeV = 0;  // GPA_HARMONIZATION
  for ( const auto& iter : unionPatchTemp ) {
    const auto& curPatchUnion = iter.second;
    occupancySizeU            = std::max<size_t>( occupancySizeU, curPatchUnion.getSizeU0() + 1 );
    occupancySizeV            = std::max<size_t>( occupancySizeV, curPatchUnion.getSizeV0() + 1 );
  }
  size_t width  = occupancySizeU * params_.occupancyResolution_;
  size_t height = occupancySizeV * params_.occupancyResolution_;
  size_t maxOccupancyRow{0};

  std::vector<bool> occupancyMap;
  int               numOrientations = params_.packingStrategy_ == 0 ? 1 : ( params_.useEightOrientations_ ? 8 : 2 );
  occupancyMap.resize( occupancySizeU * occupancySizeV, false );
  for ( auto& iter : unionPatchTemp ) {
    auto& curPatchUnion = iter.second;  // [u0, v0] may be modified;
    assert( curPatchUnion.getSizeU0() < occupancySizeU );
    assert( curPatchUnion.getSizeV0() < occupancySizeV );
    bool  locationFound = false;
    auto& occupancy     = curPatchUnion.getOccupancy();
    while ( !locationFound ) {
      for ( size_t v = 0; v < occupancySizeV && !locationFound; ++v ) {
        for ( size_t u = 0; u < occupancySizeU && !locationFound; ++u ) {
          curPatchUnion.getU0() = u;
          curPatchUnion.getV0() = v;
          if ( params_.packingStrategy_ == 0 ) {
            curPatchUnion.getPatchOrientation() = PATCH_ORIENTATION_DEFAULT;
            if ( curPatchUnion.checkFitPatchCanvas( occupancyMap, occupancySizeU, occupancySizeV,
                                                    params_.lowDelayEncoding_, safeguard ) ) {
              locationFound = true;
              if ( printDetailedInfo ) {
                std::cout << "Orientation " << curPatchUnion.getPatchOrientation() << " selected for unionPatch "
                          << curPatchUnion.getIndex() << " (" << u << "," << v << ")" << std::endl;
              }
            }
          } else {
            if ( useRefFrame && ( curPatchUnion.getPatchOrientation() != -1 ) ) {
              // already knonw Patch Orientation. just try.
              if ( curPatchUnion.checkFitPatchCanvas( occupancyMap, occupancySizeU, occupancySizeV,
                                                      params_.lowDelayEncoding_, safeguard ) ) {
                locationFound = true;
                if ( printDetailedInfo ) {
                  std::cout << "location u0,v0 selected for unionPatch " << curPatchUnion.getIndex() << " (" << u << ","
                            << v << ")" << std::endl;
                }
              }
            } else {
              for ( size_t orientationIdx = 0; orientationIdx < numOrientations && !locationFound; orientationIdx++ ) {
                if ( curPatchUnion.getSizeU0() > curPatchUnion.getSizeV0() ) {
                  curPatchUnion.getPatchOrientation() = orientation_horizontal[orientationIdx];
                } else {
                  curPatchUnion.getPatchOrientation() = orientation_vertical[orientationIdx];
                }
                if ( curPatchUnion.checkFitPatchCanvas( occupancyMap, occupancySizeU, occupancySizeV,
                                                        params_.lowDelayEncoding_, safeguard ) ) {
                  locationFound = true;
                  if ( printDetailedInfo ) {
                    std::cout << "Orientation " << curPatchUnion.getPatchOrientation() << " selected for unionPatch "
                              << curPatchUnion.getIndex() << " (" << u << "," << v << ")" << std::endl;
                  }
                }
              }
            }
          }
        }
      }
      if ( !locationFound ) {
        occupancySizeV *= 2;
        occupancyMap.resize( occupancySizeU * occupancySizeV );
      }
    }
    for ( size_t v0 = 0; v0 < curPatchUnion.getSizeV0(); ++v0 ) {
      for ( size_t u0 = 0; u0 < curPatchUnion.getSizeU0(); ++u0 ) {
        int coord = curPatchUnion.patchBlock2CanvasBlock( u0, v0, occupancySizeU, occupancySizeV );
        if ( params_.lowDelayEncoding_ ) {
          occupancyMap[coord] = true;
        } else {
          occupancyMap[coord] = occupancyMap[coord] || occupancy[v0 * curPatchUnion.getSizeU0() + u0];
        }
      }
    }

    if ( !( curPatchUnion.isPatchDimensionSwitched() ) ) {
      height =
          ( std::max )( height, ( curPatchUnion.getV0() + curPatchUnion.getSizeV0() ) * params_.occupancyResolution_ );
      width =
          ( std::max )( width, ( curPatchUnion.getU0() + curPatchUnion.getSizeU0() ) * params_.occupancyResolution_ );
      maxOccupancyRow = ( std::max )( maxOccupancyRow, ( curPatchUnion.getV0() + curPatchUnion.getSizeV0() ) );
    } else {
      height =
          ( std::max )( height, ( curPatchUnion.getV0() + curPatchUnion.getSizeU0() ) * params_.occupancyResolution_ );
      width =
          ( std::max )( width, ( curPatchUnion.getU0() + curPatchUnion.getSizeV0() ) * params_.occupancyResolution_ );
      maxOccupancyRow = ( std::max )( maxOccupancyRow, ( curPatchUnion.getV0() + curPatchUnion.getSizeU0() ) );
    }
  }
  std::cout << "\tunionPatchPacking: actualImageSize " << width << " x " << height << std::endl;
  return height;
}

void PCCEncoder::packingFirstFrame( PCCContext& context,
                                    size_t      frameIndex,
                                    size_t      tileIndex,
                                    size_t      frameWidth,
                                    size_t      frameHeight,
                                    bool        packingStrategy,
                                    int         safeguard,
                                    bool        hasRefFrame ) {
  PCCFrameContext& tile           = context[frameIndex].getTile( tileIndex );
  auto&            patches        = tile.getPatches();
  size_t           occupancySizeU = frameWidth / params_.occupancyResolution_;
  size_t           occupancySizeV = ( std::max )( patches[0].getSizeU0(), patches[0].getSizeV0() );

  for ( auto& patch : patches ) { occupancySizeU = ( std::max )( occupancySizeU, patch.getSizeU0() + 1 ); }
  auto& widthGPA = tile.getCurPCCGPAFrameSize().widthGPA_;
  auto& heithGPA = tile.getCurPCCGPAFrameSize().heightGPA_;
  widthGPA       = occupancySizeU * params_.occupancyResolution_;
  heithGPA       = occupancySizeV * params_.occupancyResolution_;
  size_t maxOccupancyRow{0};

  {
    int numOrientations = ( params_.packingStrategy_ == 0 ) ? 1 : ( params_.useEightOrientations_ ? 8 : 2 );

    std::vector<bool> occupancyMap;
    occupancyMap.resize( occupancySizeU * occupancySizeV, false );

    for ( auto& patch : patches ) {
      assert( patch.getSizeU0() <= occupancySizeU );
      assert( patch.getSizeV0() <= occupancySizeV );
      bool          locationFound   = false;
      auto&         occupancy       = patch.getOccupancy();
      GPAPatchData& curGPAPatchData = patch.getCurGPAPatchData();
      curGPAPatchData.occupancy     = occupancy;
      curGPAPatchData.sizeU0        = patch.getSizeU0();
      curGPAPatchData.sizeV0        = patch.getSizeV0();

      while ( !locationFound ) {
        // no ref patch for first frame.
        if ( ( patch.getBestMatchIdx() != InvalidPatchIndex ) && ( hasRefFrame ) ) {
          std::vector<PCCPatch>& prevPatches = context[frameIndex - 1].getTile( tileIndex ).getPatches();
          curGPAPatchData.patchOrientation   = prevPatches[patch.getBestMatchIdx()].getPatchOrientation();
          // try to place on the same position as the matched patch
          curGPAPatchData.u0 = prevPatches[patch.getBestMatchIdx()].getU0();
          curGPAPatchData.v0 = prevPatches[patch.getBestMatchIdx()].getV0();
          if ( patch.checkFitPatchCanvasForGPA( occupancyMap, occupancySizeU, occupancySizeV,
                                                params_.lowDelayEncoding_ ) ) {
            locationFound = true;
            if ( printDetailedInfo ) {
              std::cout << "Maintained orientation " << curGPAPatchData.patchOrientation
                        << " for matched patch in the same position (" << curGPAPatchData.u0 << ","
                        << curGPAPatchData.v0 << ")" << std::endl;
            }
          }
          // if the patch couldn't fit, try to fit the patch in the top left
          // position
          for ( int v = 0; v <= occupancySizeV && !locationFound; ++v ) {
            for ( int u = 0; u <= occupancySizeU && !locationFound; ++u ) {
              curGPAPatchData.u0 = u;
              curGPAPatchData.v0 = v;
              if ( patch.checkFitPatchCanvasForGPA( occupancyMap, occupancySizeU, occupancySizeV,
                                                    params_.lowDelayEncoding_, safeguard ) ) {
                locationFound = true;
                if ( printDetailedInfo ) {
                  std::cout << "Maintained orientation " << curGPAPatchData.patchOrientation << " for matched patch:("
                            << curGPAPatchData.u0 << "," << curGPAPatchData.v0 << ")" << std::endl;
                }
              }
            }
          }
        } else {
          // best effort
          for ( size_t v = 0; v < occupancySizeV && !locationFound; ++v ) {
            for ( size_t u = 0; u < occupancySizeU && !locationFound; ++u ) {
              curGPAPatchData.u0 = u;
              curGPAPatchData.v0 = v;
              for ( size_t orientationIdx = 0; orientationIdx < numOrientations && !locationFound; orientationIdx++ ) {
                if ( params_.packingStrategy_ == 0 )
                  curGPAPatchData.patchOrientation = PATCH_ORIENTATION_DEFAULT;
                else {
                  if ( curGPAPatchData.sizeU0 > curGPAPatchData.sizeV0 ) {
                    curGPAPatchData.patchOrientation = orientation_horizontal[orientationIdx];
                  } else {
                    curGPAPatchData.patchOrientation = orientation_vertical[orientationIdx];
                  }
                }
                if ( patch.checkFitPatchCanvasForGPA( occupancyMap, occupancySizeU, occupancySizeV,
                                                      params_.lowDelayEncoding_, safeguard ) ) {
                  locationFound = true;
                  if ( printDetailedInfo ) {
                    std::cout << "Orientation " << curGPAPatchData.patchOrientation << " selected for unmatched patch:("
                              << curGPAPatchData.u0 << "," << curGPAPatchData.v0 << ")" << std::endl;
                  }
                }
              }
            }
          }
        }
        if ( !locationFound ) {
          occupancySizeV *= 2;
          occupancyMap.resize( occupancySizeU * occupancySizeV );
        }
      }
      for ( size_t v0 = 0; v0 < curGPAPatchData.sizeV0; ++v0 ) {
        for ( size_t u0 = 0; u0 < curGPAPatchData.sizeU0; ++u0 ) {
          int coord = patch.patchBlock2CanvasBlockForGPA( u0, v0, occupancySizeU, occupancySizeV );
          if ( params_.lowDelayEncoding_ ) {
            occupancyMap[coord] = true;
          } else {
            occupancyMap[coord] = occupancyMap[coord] || occupancy[v0 * patch.getSizeU0() + u0];
          }
        }
      }
      if ( !( curGPAPatchData.isPatchDimensionSwitched() ) ) {
        heithGPA =
            ( std::max )( heithGPA, ( curGPAPatchData.v0 + curGPAPatchData.sizeV0 ) * patch.getOccupancyResolution() );
        widthGPA =
            ( std::max )( widthGPA, ( curGPAPatchData.u0 + curGPAPatchData.sizeU0 ) * patch.getOccupancyResolution() );
        maxOccupancyRow = ( std::max )( maxOccupancyRow, ( curGPAPatchData.v0 + curGPAPatchData.sizeV0 ) );
      } else {
        heithGPA =
            ( std::max )( heithGPA, ( curGPAPatchData.v0 + curGPAPatchData.sizeU0 ) * patch.getOccupancyResolution() );
        widthGPA =
            ( std::max )( widthGPA, ( curGPAPatchData.u0 + curGPAPatchData.sizeV0 ) * patch.getOccupancyResolution() );
        maxOccupancyRow = ( std::max )( maxOccupancyRow, ( curGPAPatchData.v0 + curGPAPatchData.sizeU0 ) );
      }
    }

    if ( !tile.getRawPointsPatches().empty() && !tile.getUseRawPointsSeparateVideo() ) {
      packRawPointsPatch( tile, occupancyMap, widthGPA, heithGPA, occupancySizeU, occupancySizeV, maxOccupancyRow );
      for ( size_t idxRawPatches = 0; idxRawPatches < tile.getRawPointsPatches().size(); idxRawPatches++ ) {
        tile.getRawPointsPatch( idxRawPatches ).preV0_ = tile.getRawPointsPatch( idxRawPatches ).v0_;
      }
    } else {
      if ( printDetailedInfo ) { printMap( occupancyMap, occupancySizeU, occupancySizeV ); }
    }
    if ( params_.enhancedOccupancyMapCode_ && !tile.getUseRawPointsSeparateVideo() ) {
      packEOMTexturePointsPatch( tile, occupancyMap, widthGPA, heithGPA, occupancySizeU, occupancySizeV,
                                 maxOccupancyRow );
    }
    if ( printDetailedInfo ) { printMap( occupancyMap, occupancySizeU, occupancySizeV ); }

    std::cout << "\tpackingFirstFrame: actualImageSize " << widthGPA << " x " << heithGPA << std::endl;
  }
}

void PCCEncoder::updatePatchInformation( PCCContext& context, size_t tileIndex, SubContext& subContext ) {
  std::cout << "The subContext is: [" << subContext.first << ", " << subContext.second << ")" << std::endl;
  for ( size_t frameIndex = subContext.first; frameIndex < subContext.second; ++frameIndex ) {
    PCCFrameContext& tile      = context[frameIndex].getTile( tileIndex );
    tile.getGlobalPatchCount() = 0;  // GPA_HARMONIZATION
    tile.getWidth()            = tile.getPrePCCGPAFrameSize().widthGPA_;
    tile.getHeight()           = tile.getPrePCCGPAFrameSize().heightGPA_;
    auto& curPatches           = tile.getPatches();
    for ( auto& curPatch : curPatches ) {
      GPAPatchData& preGPAPatchData  = curPatch.getPreGPAPatchData();
      curPatch.getSizeU0()           = preGPAPatchData.sizeU0;
      curPatch.getSizeV0()           = preGPAPatchData.sizeV0;
      curPatch.getOccupancy()        = preGPAPatchData.occupancy;
      curPatch.getU0()               = preGPAPatchData.u0;
      curPatch.getV0()               = preGPAPatchData.v0;
      curPatch.getPatchOrientation() = preGPAPatchData.patchOrientation;
      curPatch.getIsGlobalPatch()    = preGPAPatchData.isGlobalPatch;
      // GPA_HARMONIZATION Begin --------------------------------------
      if ( curPatch.getIsGlobalPatch() ) { tile.getGlobalPatchCount() = tile.getGlobalPatchCount() + 1; }
      // GPA_HARMONIZATION End ----------------------------------------
    }
    // update rawPoints patch infor.
    if ( !tile.getRawPointsPatches().empty() && !tile.getUseRawPointsSeparateVideo() ) {
      for ( size_t idxRawPatches = 0; idxRawPatches < tile.getRawPointsPatches().size(); idxRawPatches++ ) {
        // frame.getRawPointsPatch( idxRawPatches ).u0_ =
        // frame.getRawPointsPatch( idxRawPatches ).preU0_;
        tile.getRawPointsPatch( idxRawPatches ).v0_ = tile.getRawPointsPatch( idxRawPatches ).preV0_;
      }
    }
  }

  // GPA_HARMONIZATION Begin  --------------------------------------
  // no need to update single frame.
  if ( subContext.second - subContext.first == 1 ) {  // only one frame
    // Reset bestMatchIndex for the first frame.
    PCCFrameContext& tile       = context[subContext.first].getTile( tileIndex );
    auto&            curPatches = tile.getPatches();
    for ( auto& patch : curPatches ) { patch.setBestMatchIdx( InvalidPatchIndex ); }
    tile.setNumMatchedPatches( 0 );
    return;
  }
  int globalPatchCount = 0;
  for ( size_t frameIndex = subContext.first; frameIndex < subContext.second; frameIndex++ ) {
    PCCFrameContext& tile       = context[frameIndex].getTile( tileIndex );
    auto&            curPatches = tile.getPatches();
    for ( size_t index = 0; index < curPatches.size(); index++ ) { curPatches[index].getIndex() = index; }
    // reorder the patches.
    vector<PCCPatch> reorderPatches = curPatches;
    globalPatchCount                = tile.getGlobalPatchCount();
    std::cout << "\tframeIndex:" << frameIndex << " tileIndex:" << tileIndex << " patchCount:" << reorderPatches.size();
    std::cout << " frame.getGlobalPatchCount() = " << globalPatchCount << std::endl;
    fflush( stdout );

    curPatches.clear();
    curPatches.resize( 0 );
    if ( frameIndex == subContext.first ) {
      for ( auto& patch : reorderPatches ) {
        if ( patch.getIsGlobalPatch() ) { curPatches.emplace_back( patch ); }
      }
      for ( auto& patch : reorderPatches ) {
        if ( !patch.getIsGlobalPatch() ) { curPatches.emplace_back( patch ); }
      }
    } else {
      // get global patch.
      auto& prevTile = context[frameIndex - 1].getTile( tileIndex );
      for ( int32_t index = 0; index < prevTile.getPatches().size(); index++ ) {
        for ( auto& patch : reorderPatches ) {
          if ( ( index == patch.getBestMatchIdx() ) && ( patch.getIsGlobalPatch() ) ) {
            curPatches.emplace_back( patch );
            break;
          }
        }
      }
      // get non-global patch.
      for ( auto& patch : reorderPatches ) {
        if ( !patch.getIsGlobalPatch() ) { curPatches.emplace_back( patch ); }
      }
    }
  }

  // adjust index.
  for ( size_t frameIndex = subContext.first; frameIndex < subContext.second; frameIndex++ ) {
    auto& curPatches = context[frameIndex].getTile( tileIndex ).getPatches();
    // global patch.
    for ( int32_t index = 0; index < globalPatchCount; index++ ) {
      if ( frameIndex > subContext.first ) { curPatches[index].setBestMatchIdx( index ); }
      curPatches[index].getIndex() = index;
    }
    if ( frameIndex == subContext.second - 1 ) {
      for ( int32_t index = globalPatchCount; index < curPatches.size(); index++ ) {
        curPatches[index].getIndex() = index;
      }
      continue;
    }

    // non-global patches.
    auto&             nextPatches = context[frameIndex + 1].getTile( tileIndex ).getPatches();
    std::vector<bool> updated;
    updated.resize( nextPatches.size(), false );
    std::cout << "\treference Patch Idx[frame:patchIdx/patchCout,referencePatchIdx] :\n";
    for ( int32_t index = globalPatchCount; index < curPatches.size(); index++ ) {
      for ( int32_t i = globalPatchCount; i < nextPatches.size(); i++ ) {
        if ( ( static_cast<int32_t>( curPatches[index].getIndex() ) == nextPatches[i].getBestMatchIdx() ) &&
             ( !updated[i] ) ) {
          nextPatches[i].setBestMatchIdx( index );
          updated[i] = true;
          break;
        }
      }
      curPatches[index].getIndex() = static_cast<size_t>( index );
      std::cout << "\t[" << frameIndex << ":" << index << "/" << curPatches.size() << ", "
                << curPatches[index].getBestMatchIdx() << "] ";
      fflush( stdout );
    }
    std::cout << std::endl;
  }

  // Reset bestMatchIndex for the first frame.
  PCCFrameContext& tile       = context[subContext.first].getTile( tileIndex );
  auto&            curPatches = tile.getPatches();
  for ( auto& patch : curPatches ) { patch.setBestMatchIdx( InvalidPatchIndex ); }
  tile.setNumMatchedPatches( 0 );

  // Other frames in the current subContext
  for ( size_t frameIndex = subContext.first + 1; frameIndex < subContext.second; frameIndex++ ) {
    auto&  curPatches        = context[frameIndex].getTile( tileIndex ).getPatches();
    size_t numMatchedPatches = globalPatchCount;
    for ( int32_t index = globalPatchCount; index < curPatches.size(); index++ ) {
      if ( curPatches[index].getBestMatchIdx() != InvalidPatchIndex ) { numMatchedPatches++; }
    }
    context[frameIndex].getTile( tileIndex ).setNumMatchedPatches( numMatchedPatches );
  }

  // GPA_HARMONIZATION End ----------------------------------------
}

void PCCEncoder::updateGPAPatchInformation( PCCContext& context,
                                            size_t      tileIndex,
                                            SubContext& subContext,
                                            unionPatch& unionPatch ) {
  for ( size_t i = subContext.first; i < subContext.second; ++i ) {
    auto& patches = context[i].getTile( tileIndex ).getPatches();
    for ( auto& patch : patches ) {
      GPAPatchData& curGPAPatchData = patch.getCurGPAPatchData();
      if ( curGPAPatchData.isGlobalPatch ) {
        size_t            globalIndex      = curGPAPatchData.globalPatchIndex;
        auto&             cPatchUnion      = unionPatch[globalIndex];
        size_t            initialSizeU0    = patch.getSizeU0();
        size_t            initialSizeV0    = patch.getSizeV0();
        size_t            updatedSizeU0    = cPatchUnion.getSizeU0();
        size_t            updatedSizeV0    = cPatchUnion.getSizeV0();
        auto&             initialOccupancy = patch.getOccupancy();
        std::vector<bool> updatedOccupancy( updatedSizeU0 * updatedSizeV0, false );
        for ( size_t v0 = 0; v0 < initialSizeV0; ++v0 ) {
          for ( size_t u0 = 0; u0 < initialSizeU0; ++u0 ) {
            size_t initialIndex = v0 * initialSizeU0 + u0;
            size_t updatedIndex = v0 * updatedSizeU0 + u0;
            if ( initialOccupancy[initialIndex] && !updatedOccupancy[updatedIndex] ) {
              updatedOccupancy[updatedIndex] = true;
            }
          }
        }
        curGPAPatchData.sizeU0    = updatedSizeU0;
        curGPAPatchData.sizeV0    = updatedSizeV0;
        curGPAPatchData.occupancy = updatedOccupancy;
      } else {
        curGPAPatchData.sizeU0    = patch.getSizeU0();
        curGPAPatchData.sizeV0    = patch.getSizeV0();
        curGPAPatchData.occupancy = patch.getOccupancy();
      }
    }
  }
}

void PCCEncoder::performGPAPacking( const SubContext& subContext,
                                    unionPatch&       unionPatch,
                                    PCCContext&       context,
                                    size_t            tileIndex,
                                    size_t            frameWidth,
                                    size_t            frameHeight,
                                    bool&             badGPAPacking,
                                    size_t            unionsHeight,
                                    int               safeguard,
                                    bool              useRefFrame ) {
  bool   exceedMinimumImageHeight = false;  // whether exceed minimunImageHeight or not;
  size_t badCondition             = 0;      // GPA bad condition count;
  for ( size_t i = subContext.first; i < subContext.second; ++i ) {
    auto& curFrameContext = context[i].getTile( tileIndex );
    auto& widthGPA        = curFrameContext.getCurPCCGPAFrameSize().widthGPA_;
    auto& heightGPA       = curFrameContext.getCurPCCGPAFrameSize().heightGPA_;
    auto& patches         = curFrameContext.getPatches();
    if ( patches.empty() ) { return; }
    int   preIndex   = i > 0 ? i - 1 : 0;
    auto& prePatches = context[preIndex].getTile( tileIndex ).getPatches();

    size_t occupancySizeU = frameWidth / params_.occupancyResolution_;
    size_t occupancySizeV = unionsHeight / params_.occupancyResolution_;
    for ( auto& patch : patches ) {
      occupancySizeU = ( std::max )( occupancySizeU, patch.getCurGPAPatchData().sizeU0 + 1 );
    }
    widthGPA  = occupancySizeU * params_.occupancyResolution_;
    heightGPA = occupancySizeV * params_.occupancyResolution_;
    size_t            maxOccupancyRow{0};
    std::vector<bool> occupancyMap;
    occupancyMap.resize( occupancySizeU * occupancySizeV, false );
    // !!!packing global matched patch;
    for ( auto& patch : patches ) {
      GPAPatchData& curGPAPatchData = patch.getCurGPAPatchData();
      if ( curGPAPatchData.isGlobalPatch ) {
        assert( curGPAPatchData.sizeU0 <= occupancySizeU );
        assert( curGPAPatchData.sizeV0 <= occupancySizeV );
        const size_t trackIndex = curGPAPatchData.globalPatchIndex;
        assert( unionPatch.count( trackIndex ) != 0 );
        curGPAPatchData.u0               = unionPatch[trackIndex].getU0();
        curGPAPatchData.v0               = unionPatch[trackIndex].getV0();
        curGPAPatchData.patchOrientation = unionPatch[trackIndex].getPatchOrientation();
        if ( printDetailedInfo ) {
          std::cout << "Orientation:" << curGPAPatchData.patchOrientation << " for GPA patch in the same position ("
                    << curGPAPatchData.u0 << "," << curGPAPatchData.v0 << ") size: " << curGPAPatchData.sizeU0 << "x"
                    << curGPAPatchData.sizeV0 << " tileSize " << curFrameContext.getWidth() << "x"
                    << curFrameContext.getHeight() << std::endl;
        }
        for ( size_t v0 = 0; v0 < curGPAPatchData.sizeV0; ++v0 ) {
          for ( size_t u0 = 0; u0 < curGPAPatchData.sizeU0; ++u0 ) {
            int coord = patch.patchBlock2CanvasBlockForGPA( u0, v0, occupancySizeU, occupancySizeV );
            if ( params_.lowDelayEncoding_ ) {
              occupancyMap[coord] = true;
            } else {
              occupancyMap[coord] = occupancyMap[coord] || curGPAPatchData.occupancy[v0 * curGPAPatchData.sizeU0 + u0];
            }
          }
        }
        if ( !( curGPAPatchData.isPatchDimensionSwitched() ) ) {
          heightGPA       = ( std::max )( heightGPA,
                                    ( curGPAPatchData.v0 + curGPAPatchData.sizeV0 ) * patch.getOccupancyResolution() );
          widthGPA        = ( std::max )( widthGPA,
                                   ( curGPAPatchData.u0 + curGPAPatchData.sizeU0 ) * patch.getOccupancyResolution() );
          maxOccupancyRow = ( std::max )( maxOccupancyRow, ( curGPAPatchData.v0 + curGPAPatchData.sizeV0 ) );
        } else {
          heightGPA       = ( std::max )( heightGPA,
                                    ( curGPAPatchData.v0 + curGPAPatchData.sizeU0 ) * patch.getOccupancyResolution() );
          widthGPA        = ( std::max )( widthGPA,
                                   ( curGPAPatchData.u0 + curGPAPatchData.sizeV0 ) * patch.getOccupancyResolution() );
          maxOccupancyRow = ( std::max )( maxOccupancyRow, ( curGPAPatchData.v0 + curGPAPatchData.sizeU0 ) );
        }
      }
    }
    // !!!packing non-global matched patch;
    int icount = -1;
    for ( auto& patch : patches ) {
      icount++;
      GPAPatchData& curGPAPatchData = patch.getCurGPAPatchData();
      if ( curGPAPatchData.isGlobalPatch ) { continue; }

      // not use reference frame only if the first frame or useRefFrame is
      // disabled.
      if ( ( i == 0 ) || ( ( i == subContext.first ) && ( !useRefFrame ) ) ) {  // not use ref.
        packingWithoutRefForFirstFrameNoglobalPatch( patch, i, icount, occupancySizeU, occupancySizeV, safeguard,
                                                     occupancyMap, heightGPA, widthGPA, maxOccupancyRow );
      } else {
        // PCCPatch prePatch = prePatches[patch.getBestMatchIdx()];
        packingWithRefForFirstFrameNoglobalPatch( patch, prePatches, subContext.first, i, icount, occupancySizeU,
                                                  occupancySizeV, safeguard, occupancyMap, heightGPA, widthGPA,
                                                  maxOccupancyRow );
      }
    }

    if ( !curFrameContext.getRawPointsPatches().empty() && !curFrameContext.getUseRawPointsSeparateVideo() ) {
      packRawPointsPatch( curFrameContext, occupancyMap, widthGPA, heightGPA, occupancySizeU, occupancySizeV,
                          maxOccupancyRow );
      for ( size_t idxRawPatches = 0; idxRawPatches < curFrameContext.getRawPointsPatches().size(); idxRawPatches++ ) {
        curFrameContext.getRawPointsPatch( idxRawPatches ).tempV0_ =
            curFrameContext.getRawPointsPatch( idxRawPatches ).v0_;
      }
    } else {
      if ( printDetailedInfo ) { printMap( occupancyMap, occupancySizeU, occupancySizeV ); }
    }
    if ( params_.enhancedOccupancyMapCode_ && !curFrameContext.getUseRawPointsSeparateVideo() ) {
      packEOMTexturePointsPatch( curFrameContext, occupancyMap, widthGPA, heightGPA, occupancySizeU, occupancySizeV,
                                 maxOccupancyRow );
    }
    if ( printDetailedInfo ) { printMap( occupancyMap, occupancySizeU, occupancySizeV ); }

    // determination......;
    if ( heightGPA > params_.minimumImageHeight_ ) {
      exceedMinimumImageHeight = true;
      break;
    }
    double validHeightRatio = ( double( heightGPA ) ) / ( double( curFrameContext.getHeight() ) );
    if ( validHeightRatio >= BAD_HEIGHT_THRESHOLD ) { badCondition++; }
  }

  if ( exceedMinimumImageHeight || badCondition > BAD_CONDITION_THRESHOLD ) { badGPAPacking = true; }
}

void PCCEncoder::packingWithoutRefForFirstFrameNoglobalPatch(
    PCCPatch&          patch,
    size_t             ii,
    size_t             icount,
    size_t&            occupancySizeU,
    size_t&            occupancySizeV,
    const size_t       safeguard,
    std::vector<bool>& occupancyMap,
    size_t&            heightGPA,
    size_t&            widthGPA,
    size_t&            maxOccupancyRow ) {  // GPA_HAMONIZATION, the whole function has been
                                 // changed
  int           numOrientations = ( params_.packingStrategy_ == 0 ) ? 1 : ( params_.useEightOrientations_ ? 8 : 2 );
  GPAPatchData& curGPAPatchData = patch.getCurGPAPatchData();  // GPAPatchData& preGPAPatchData =
                                                               // patch.getCurGPAPatchData();

  assert( curGPAPatchData.sizeU0 <= occupancySizeU );
  assert( curGPAPatchData.sizeV0 <= occupancySizeV );
  bool  locationFound = false;
  auto& occupancy     = patch.getOccupancy();
  while ( !locationFound ) {
    for ( size_t v = 0; v < occupancySizeV && !locationFound; ++v ) {
      for ( size_t u = 0; u < occupancySizeU && !locationFound; ++u ) {
        curGPAPatchData.u0 = u;
        curGPAPatchData.v0 = v;
        if ( params_.packingStrategy_ == 0 ) {
          curGPAPatchData.patchOrientation = PATCH_ORIENTATION_DEFAULT;
          // std::cout<<"checkFitPatchCanvasForGPA"<<std::endl;
          if ( patch.checkFitPatchCanvasForGPA( occupancyMap, occupancySizeU, occupancySizeV, params_.lowDelayEncoding_,
                                                safeguard ) ) {
            locationFound = true;
            if ( printDetailedInfo ) {
              std::cout << "Orientation " << curGPAPatchData.patchOrientation << " selected for Patch: ["
                        << icount  // preGPAPatchData->curXXXX
                        << "] in the position (" << u << "," << v << ")" << std::endl;
            }
          }
        } else {  // try several orientation.
          for ( size_t orientationIdx = 0; orientationIdx < numOrientations && !locationFound; orientationIdx++ ) {
            if ( params_.packingStrategy_ == 0 )
              curGPAPatchData.patchOrientation = PATCH_ORIENTATION_DEFAULT;
            else {
              if ( patch.getSizeU0() > patch.getSizeV0() ) {
                curGPAPatchData.patchOrientation = orientation_horizontal[orientationIdx];
              } else {
                curGPAPatchData.patchOrientation = orientation_vertical[orientationIdx];
              }
            }
            if ( patch.checkFitPatchCanvasForGPA( occupancyMap, occupancySizeU, occupancySizeV,
                                                  params_.lowDelayEncoding_, safeguard ) ) {
              locationFound = true;
              if ( printDetailedInfo ) {
                std::cout << "Orientation " << curGPAPatchData.patchOrientation << "selected for Patch: [" << icount
                          << "] in the position (" << curGPAPatchData.u0 << "," << curGPAPatchData.v0 << ")"
                          << std::endl;
              }
            }
          }
        }
      }
    }
    if ( !locationFound ) {
      occupancySizeV *= 2;
      occupancyMap.resize( occupancySizeU * occupancySizeV );
      if ( printDetailedInfo ) { std::cout << "Increase occupancySizeV " << occupancySizeV << std::endl; }
    }
  }
  // update occupancy.
  for ( size_t v0 = 0; v0 < curGPAPatchData.sizeV0; ++v0 ) {
    for ( size_t u0 = 0; u0 < curGPAPatchData.sizeU0; ++u0 ) {
      int coord = patch.patchBlock2CanvasBlockForGPA( u0, v0, occupancySizeU, occupancySizeV );
      if ( params_.lowDelayEncoding_ ) {
        occupancyMap[coord] = true;
      } else {
        occupancyMap[coord] = occupancyMap[coord] || occupancy[v0 * patch.getSizeU0() + u0];
      }
    }
  }
  if ( !( curGPAPatchData.isPatchDimensionSwitched() ) ) {
    heightGPA =
        ( std::max )( heightGPA, ( curGPAPatchData.v0 + curGPAPatchData.sizeV0 ) * patch.getOccupancyResolution() );
    widthGPA =
        ( std::max )( widthGPA, ( curGPAPatchData.u0 + curGPAPatchData.sizeU0 ) * patch.getOccupancyResolution() );
    maxOccupancyRow = ( std::max )( maxOccupancyRow, ( curGPAPatchData.v0 + curGPAPatchData.sizeV0 ) );
  } else {
    heightGPA =
        ( std::max )( heightGPA, ( curGPAPatchData.v0 + curGPAPatchData.sizeU0 ) * patch.getOccupancyResolution() );
    widthGPA =
        ( std::max )( widthGPA, ( curGPAPatchData.u0 + curGPAPatchData.sizeV0 ) * patch.getOccupancyResolution() );
    maxOccupancyRow = ( std::max )( maxOccupancyRow, ( curGPAPatchData.v0 + curGPAPatchData.sizeU0 ) );
  }
}
void PCCEncoder::packingWithRefForFirstFrameNoglobalPatch( PCCPatch&                    patch,
                                                           const std::vector<PCCPatch>& prePatches,
                                                           size_t                       startFrameIndex,
                                                           size_t                       i,
                                                           size_t                       icount,
                                                           size_t&                      occupancySizeU,
                                                           size_t&                      occupancySizeV,
                                                           const size_t                 safeguard,
                                                           std::vector<bool>&           occupancyMap,
                                                           size_t&                      heightGPA,
                                                           size_t&                      widthGPA,
                                                           size_t&                      maxOccupancyRow ) {
  // favoring horizontal orientations (that should be rotated)
  int32_t       numOrientations = ( params_.packingStrategy_ == 0 ) ? 1 : ( params_.useEightOrientations_ ? 8 : 2 );
  GPAPatchData& curGPAPatchData = patch.getCurGPAPatchData();  // GPA_HARMONIZATION

  assert( curGPAPatchData.sizeU0 <= occupancySizeU );
  assert( curGPAPatchData.sizeV0 <= occupancySizeV );
  bool  locationFound = false;
  auto& occupancy     = patch.getOccupancy();
  while ( !locationFound ) {
    if ( patch.getBestMatchIdx() != InvalidPatchIndex ) {
      PCCPatch prePatch = prePatches[patch.getBestMatchIdx()];
      if ( i == startFrameIndex ) {
        curGPAPatchData.patchOrientation = prePatch.getPatchOrientation();
        // try to place on the same position as the matched patch
        curGPAPatchData.u0 = prePatch.getU0();
        curGPAPatchData.v0 = prePatch.getV0();
      } else {
        curGPAPatchData.patchOrientation = prePatch.getCurGPAPatchData().patchOrientation;
        // try to place on the same position as the matched patch
        curGPAPatchData.u0 = prePatch.getCurGPAPatchData().u0;
        curGPAPatchData.v0 = prePatch.getCurGPAPatchData().v0;
      }
      if ( curGPAPatchData.patchOrientation == -1 ) { assert( curGPAPatchData.patchOrientation != -1 ); }

      if ( patch.checkFitPatchCanvasForGPA( occupancyMap, occupancySizeU, occupancySizeV, params_.lowDelayEncoding_,
                                            safeguard ) ) {
        locationFound = true;
        if ( printDetailedInfo ) {
          std::cout << "Maintained TempGPA.orientation " << curGPAPatchData.patchOrientation << " for patch[" << icount
                    << "] in the same position (" << curGPAPatchData.u0 << "," << curGPAPatchData.v0 << ")"
                    << std::endl;
        }
      }

      // if the patch couldn't fit, try to fit the patch in the top left
      // position
      for ( int v = 0; v <= occupancySizeV && !locationFound; ++v ) {
        for ( int u = 0; u <= occupancySizeU && !locationFound; ++u ) {
          curGPAPatchData.u0 = u;
          curGPAPatchData.v0 = v;
          if ( patch.checkFitPatchCanvasForGPA( occupancyMap, occupancySizeU, occupancySizeV, params_.lowDelayEncoding_,
                                                safeguard ) ) {  // !!! function overload for GPA;
            locationFound = true;
            if ( printDetailedInfo ) {
              std::cout << "Maintained TempGPA.orientation " << curGPAPatchData.patchOrientation
                        << " for unmatched patch[" << icount << "] in the position (" << curGPAPatchData.u0 << ","
                        << curGPAPatchData.v0 << ")" << std::endl;
            }
          }
        }
      }
    } else {
      // best effort
      for ( size_t v = 0; v < occupancySizeV && !locationFound; ++v ) {
        for ( size_t u = 0; u < occupancySizeU && !locationFound; ++u ) {
          curGPAPatchData.u0 = u;
          curGPAPatchData.v0 = v;
          for ( size_t orientationIdx = 0; orientationIdx < numOrientations && !locationFound; orientationIdx++ ) {
            if ( params_.packingStrategy_ == 0 ) curGPAPatchData.patchOrientation = PATCH_ORIENTATION_DEFAULT;
            {
              if ( patch.getSizeU0() > patch.getSizeV0() ) {
                curGPAPatchData.patchOrientation = orientation_horizontal[orientationIdx];
              } else {
                curGPAPatchData.patchOrientation = orientation_vertical[orientationIdx];
              }
            }
            if ( patch.checkFitPatchCanvasForGPA( occupancyMap, occupancySizeU, occupancySizeV,
                                                  params_.lowDelayEncoding_, safeguard ) ) {
              locationFound = true;
              if ( printDetailedInfo ) {
                std::cout << "Maintained TempGPA.orientation " << curGPAPatchData.patchOrientation
                          << " for unmatched patch[" << icount << "] in the position (" << curGPAPatchData.u0 << ","
                          << curGPAPatchData.v0 << ")" << std::endl;
              }
            }
          }
        }
      }
    }
    if ( !locationFound ) {
      occupancySizeV *= 2;
      occupancyMap.resize( occupancySizeU * occupancySizeV );
      if ( printDetailedInfo ) { std::cout << "Increase occupancySizeV " << occupancySizeV << std::endl; }
    }
  }
  for ( size_t v0 = 0; v0 < curGPAPatchData.sizeV0; ++v0 ) {
    for ( size_t u0 = 0; u0 < curGPAPatchData.sizeU0; ++u0 ) {
      int coord = patch.patchBlock2CanvasBlockForGPA( u0, v0, occupancySizeU, occupancySizeV );
      if ( params_.lowDelayEncoding_ ) {
        occupancyMap[coord] = true;
      } else {
        occupancyMap[coord] = occupancyMap[coord] || occupancy[v0 * curGPAPatchData.sizeU0 + u0];
      }
    }
  }
  if ( !( curGPAPatchData.isPatchDimensionSwitched() ) ) {
    heightGPA =
        ( std::max )( heightGPA, ( curGPAPatchData.v0 + curGPAPatchData.sizeV0 ) * patch.getOccupancyResolution() );
    widthGPA =
        ( std::max )( widthGPA, ( curGPAPatchData.u0 + curGPAPatchData.sizeU0 ) * patch.getOccupancyResolution() );
    maxOccupancyRow = ( std::max )( maxOccupancyRow, ( curGPAPatchData.v0 + curGPAPatchData.sizeV0 ) );
  } else {
    heightGPA =
        ( std::max )( heightGPA, ( curGPAPatchData.v0 + curGPAPatchData.sizeU0 ) * patch.getOccupancyResolution() );
    widthGPA =
        ( std::max )( widthGPA, ( curGPAPatchData.u0 + curGPAPatchData.sizeV0 ) * patch.getOccupancyResolution() );
    maxOccupancyRow = ( std::max )( maxOccupancyRow, ( curGPAPatchData.v0 + curGPAPatchData.sizeU0 ) );
  }
}

void PCCEncoder::setPointLocalReconstruction( PCCContext& context ) {
  TRACE_PATCH( "setPointLocalReconstruction \n" );
  auto& asps = context.getAtlasSequenceParameterSet( 0 );
  asps.setPLREnabledFlag( true );
  asps.allocatePLRInformation();
  auto& plri = asps.getPLRInformation( 0 );
  plri.setMapEnabledFlag( true );
  plri.setNumberOfModesMinus1( params_.plrlNumberOfModes_ - 1 );
  plri.setBlockThresholdPerPatchMinus1( params_.patchSize_ - 1 );
  plri.allocate();
  for ( size_t i = 0; i < plri.getNumberOfModesMinus1() + 1; i++ ) {
    auto& mode = context.getPointLocalReconstructionMode( i + 1 );
    plri.setInterpolateFlag( i, mode.interpolate_ );
    plri.setFillingFlag( i, mode.filling_ );
    plri.setMinimumDepth( i, mode.minD1_ );
    plri.setNeighbourMinus1( i, mode.neighbor_ - 1 );
  }
#ifdef CODEC_TRACE
  for ( size_t i = 0; i < context.getPointLocalReconstructionModeNumber(); i++ ) {
    auto& mode = context.getPointLocalReconstructionMode( i );
    TRACE_PATCH( "Plrm[%u]: Inter = %d Fill = %d minD1 = %u neighbor = %u \n", i, mode.interpolate_, mode.filling_,
                 mode.minD1_, mode.neighbor_ );
  }
#endif
}

void PCCEncoder::setPLRData( PCCFrameContext& frame,
                             const PCCPatch&  patch,
                             PLRData&         plrd,
                             size_t           occupancyPackingBlockSize,
                             size_t           patchIndex ) {
  plrd.allocate( patch.getSizeU0(), patch.getSizeV0() );
  const size_t blockToPatchWidth  = frame.getWidth() / params_.occupancyResolution_;
  const size_t blockToPatchHeight = frame.getHeight() / params_.occupancyResolution_;
  TRACE_PATCH( "WxH = %zu x %zu \n", plrd.getBlockToPatchMapWidth(), plrd.getBlockToPatchMapHeight() );
  plrd.setLevelFlag( patch.getPointLocalReconstructionLevel() != 0u );
  TRACE_PATCH( "  LevelFlag = %d \n", plrd.getLevelFlag() );
  if ( plrd.getLevelFlag() ) {
    plrd.setPresentFlag( patch.getPointLocalReconstructionMode() > 0 );
    plrd.setModeMinus1( patch.getPointLocalReconstructionMode() - 1 );
    TRACE_PATCH( "  ModePatch: Present = %d ModeMinus1 = %2d \n", plrd.getPresentFlag(),
                 plrd.getPresentFlag() ? (int32_t)plrd.getModeMinus1() : -1 );
  } else {
    auto& blockToPatch = frame.getBlockToPatch();
    for ( size_t v0 = 0; v0 < plrd.getBlockToPatchMapHeight(); ++v0 ) {
      for ( size_t u0 = 0; u0 < plrd.getBlockToPatchMapWidth(); ++u0 ) {
        size_t index = v0 * plrd.getBlockToPatchMapWidth() + u0;
        int    pos   = patch.patchBlock2CanvasBlock( ( u0 ), ( v0 ), blockToPatchWidth, blockToPatchHeight );
        bool   occupied =
            ( blockToPatch[pos] == patchIndex + 1 ) && ( patch.getPointLocalReconstructionMode( u0, v0 ) > 0 );
        plrd.setBlockPresentFlag( index, occupied );
        if ( occupied ) { plrd.setBlockModeMinus1( index, patch.getPointLocalReconstructionMode( u0, v0 ) - 1 ); }
        TRACE_PATCH( "  Mode[%3u]: Present = %d ModeMinus1 = %2d \n", index, plrd.getBlockPresentFlag( index ),
                     plrd.getBlockPresentFlag( index ) ? (int32_t)plrd.getBlockModeMinus1( index ) : -1 );
      }
    }
  }
#ifdef CODEC_TRACE
  for ( size_t v0 = 0; v0 < patch.getSizeV0(); ++v0 ) {
    for ( size_t u0 = 0; u0 < patch.getSizeU0(); ++u0 ) {
      TRACE_PATCH(
          "Block[ %2lu %2lu <=> %4zu ] / [ %2lu %2lu ]: Level = %d Present = "
          "%d mode = %zu \n",
          u0, v0, v0 * patch.getSizeU0() + u0, patch.getSizeU0(), patch.getSizeV0(),
          patch.getPointLocalReconstructionLevel(), plrd.getBlockPresentFlag( v0 * patch.getSizeU0() + u0 ),
          patch.getPointLocalReconstructionMode( u0, v0 ) );
    }
  }
#endif
}
void PCCEncoder::setPostProcessingSeiParameters( GeneratePointCloudParameters& params, PCCContext& context ) {
  params.occupancyResolution_        = params_.occupancyResolution_;
  params.occupancyPrecision_         = params_.occupancyPrecision_;
  params.enableSizeQuantization_     = context.getEnablePatchSizeQuantization();
  params.flagGeometrySmoothing_      = params_.flagGeometrySmoothing_;
  params.gridSmoothing_              = params_.gridSmoothing_;
  params.gridSize_                   = params_.gridSize_;
  params.neighborCountSmoothing_     = params_.neighborCountSmoothing_;
  params.radius2Smoothing_           = params_.radius2Smoothing_;
  params.radius2BoundaryDetection_   = params_.radius2BoundaryDetection_;
  params.thresholdSmoothing_         = params_.thresholdSmoothing_;
  params.rawPointColorFormat_        = size_t( params_.losslessGeo444_ ? COLOURFORMAT444 : COLOURFORMAT420 );
  params.nbThread_                   = params_.nbThread_;
  params.absoluteD1_                 = params_.absoluteD1_;
  params.multipleStreams_            = params_.multipleStreams_;
  params.surfaceThickness_           = params_.surfaceThickness_;
  params.thresholdColorSmoothing_    = params_.thresholdColorSmoothing_;
  params.thresholdColorDifference_   = params_.thresholdColorDifference_;
  params.thresholdColorVariation_    = params_.thresholdColorVariation_;
  params.flagColorSmoothing_         = params_.flagColorSmoothing_;
  params.cgridSize_                  = params_.cgridSize_;
  params.enhancedOccupancyMapCode_   = params_.enhancedOccupancyMapCode_;
  params.thresholdLossyOM_           = params_.thresholdLossyOM_;
  params.removeDuplicatePoints_      = params_.removeDuplicatePoints_;
  params.pointLocalReconstruction_   = params_.pointLocalReconstruction_;
  params.mapCountMinus1_             = params_.mapCountMinus1_;
  params.singleMapPixelInterleaving_ = params_.singleMapPixelInterleaving_;
  params.geometry3dCoordinatesBitdepth_ =
      params_.geometry3dCoordinatesBitdepth_ + ( params_.additionalProjectionPlaneMode_ > 0 );
  params.useAdditionalPointsPatch_ = params_.losslessGeo_ || params_.lossyRawPointsPatch_;
  params.plrlNumberOfModes_        = params_.plrlNumberOfModes_;
  params.geometryBitDepth3D_ = params_.geometry3dCoordinatesBitdepth_ + ( params_.additionalProjectionPlaneMode_ > 0 );
  params.EOMFixBitCount_     = params_.EOMFixBitCount_;
  params.pbfEnableFlag_      = params_.pbfEnableFlag_;
  params.pbfPassesCount_     = params_.pbfPassesCount_;
  params.pbfFilterSize_      = params_.pbfFilterSize_;
  params.pbfLog2Threshold_   = params_.pbfLog2Threshold_;
}

void PCCEncoder::setGeneratePointCloudParameters( GeneratePointCloudParameters& params, PCCContext& context ) {
  params.occupancyResolution_        = params_.occupancyResolution_;
  params.occupancyPrecision_         = params_.occupancyPrecision_;
  params.enableSizeQuantization_     = context.getEnablePatchSizeQuantization();
  params.flagGeometrySmoothing_      = params_.flagGeometrySmoothing_;
  params.gridSmoothing_              = params_.gridSmoothing_;
  params.gridSize_                   = params_.gridSize_;
  params.neighborCountSmoothing_     = params_.neighborCountSmoothing_;
  params.radius2Smoothing_           = params_.radius2Smoothing_;
  params.radius2BoundaryDetection_   = params_.radius2BoundaryDetection_;
  params.thresholdSmoothing_         = params_.thresholdSmoothing_;
  params.rawPointColorFormat_        = size_t( params_.losslessGeo444_ ? COLOURFORMAT444 : COLOURFORMAT420 );
  params.nbThread_                   = params_.nbThread_;
  params.absoluteD1_                 = params_.absoluteD1_;
  params.multipleStreams_            = params_.multipleStreams_;
  params.surfaceThickness_           = params_.surfaceThickness_;
  params.thresholdColorSmoothing_    = params_.thresholdColorSmoothing_;
  params.thresholdColorDifference_   = params_.thresholdColorDifference_;
  params.thresholdColorVariation_    = params_.thresholdColorVariation_;
  params.flagColorSmoothing_         = params_.flagColorSmoothing_;
  params.cgridSize_                  = params_.cgridSize_;
  params.enhancedOccupancyMapCode_   = params_.enhancedOccupancyMapCode_;
  params.useAuxSeperateVideo_        = params_.useRawPointsSeparateVideo_;
  params.thresholdLossyOM_           = params_.thresholdLossyOM_;
  params.removeDuplicatePoints_      = params_.removeDuplicatePoints_;
  params.pointLocalReconstruction_   = params_.pointLocalReconstruction_;
  params.mapCountMinus1_             = params_.mapCountMinus1_;
  params.singleMapPixelInterleaving_ = params_.singleMapPixelInterleaving_;
  params.geometry3dCoordinatesBitdepth_ =
      params_.geometry3dCoordinatesBitdepth_ + ( params_.additionalProjectionPlaneMode_ > 0 );
  params.useAdditionalPointsPatch_ = params_.losslessGeo_ || params_.lossyRawPointsPatch_;
  params.plrlNumberOfModes_        = params_.plrlNumberOfModes_;
  params.geometryBitDepth3D_ = params_.geometry3dCoordinatesBitdepth_ + ( params_.additionalProjectionPlaneMode_ > 0 );
  params.EOMFixBitCount_     = params_.EOMFixBitCount_;
  params.pbfEnableFlag_      = false;
  params.pbfPassesCount_     = 0;
  params.pbfFilterSize_      = 0;
  params.pbfLog2Threshold_   = 0;
}

void PCCEncoder::createPatchFrameDataStructure( PCCContext& context ) {
  printf( "createPatchFrameDataStructure \n" );
  TRACE_PATCH( "createPatchFrameDataStructure GOP start \n" );
  size_t frameCount = context.getFrames().size();
  TRACE_PATCH( "frameCount = %u \n", frameCount );
  TRACE_PATCH( "PLR = %d \n", params_.pointLocalReconstruction_ );
  if ( params_.pointLocalReconstruction_ ) { setPointLocalReconstruction( context ); }

  // patch reordering
  if ( params_.patchPrecedenceOrderFlag_ ) {
    std::cout << "encoder reverse ordering" << std::endl;
    for ( size_t frameIdx = 0; frameIdx < frameCount; frameIdx++ ) {
      for ( size_t tileIdx = 0; tileIdx < context[frameIdx].getNumTilesInAtlasFrame(); tileIdx++ ) {
        auto&            patches = context[frameIdx].getTile( tileIdx ).getPatches();
        vector<PCCPatch> reverseOrderPatchList;

        for ( int i = static_cast<int>( patches.size() ) - 1; i >= 0; i-- ) {
          if ( patches[i].getBestMatchIdx() != -1 ) {
            // only 1 previous frame
            assert( frameIdx > 0 );
            patches[i].setBestMatchIdx( context[frameIdx - 1].getTile( tileIdx ).getPatches().size() -
                                        patches[i].getBestMatchIdx() - 1 );
          }

          reverseOrderPatchList.push_back( patches[i] );
        }
        patches = reverseOrderPatchList;
      }
    }
  }

  for ( size_t i = 0; i < frameCount; i++ ) {
    size_t atlasFrameParameterSetId = 0;
    // partition information
    if ( i == 0 ) {
      AtlasFrameTileInformation& afti = context.getAtlasFrameParameterSet( 0 ).getAtlasFrameTileInformation();
      generateAfti( context, 0, afti );
    } else {
      AtlasFrameTileInformation aftiUpdated;
      bool                      bPersistance = false;
      generateAfti( context, i, aftiUpdated );
      for ( size_t afpsId = 0; afpsId < context.getAtlasFrameParameterSetList().size(); afpsId++ ) {
        if ( aftiUpdated == context.getAtlasFrameParameterSet( afpsId ).getAtlasFrameTileInformation() ) {
          atlasFrameParameterSetId = afpsId;
          bPersistance             = true;
          break;
        }
      }
      if ( bPersistance == false ) {
        context.addAtlasFrameParameterSet( context.getAtlasFrameParameterSet( 0 ) ) // ajt::why update afti of afps 0?
            .setAtlasFrameTileInformation( aftiUpdated );
        atlasFrameParameterSetId = context.getAtlasFrameParameterSetList().size() - 1;
      }
    }
    for ( size_t ti = 0; ti < context[i].getNumTilesInAtlasFrame(); ti++ ) {
      auto& atl = context.addAtlasTileLayer( i, ti );
      auto& ath = atl.getHeader();
      ath.setAtlasFrameParameterSetId( atlasFrameParameterSetId );
      auto& afps = context.getAtlasFrameParameterSet( atlasFrameParameterSetId );
      // tile header
      if ( params_.additionalProjectionPlaneMode_ > 0 ) {
        ath.setPosMinDQuantizer( uint8_t( std::log2( params_.minLevel_ ) ) - 1 );
      } else {
        ath.setPosMinDQuantizer( uint8_t( std::log2( params_.minLevel_ ) ) );
      }
      ath.setPosDeltaMaxDQuantizer( uint8_t( std::log2( params_.minLevel_ ) ) );
      ath.setPatchSizeXinfoQuantizer( params_.log2QuantizerSizeX_ );
      ath.setPatchSizeYinfoQuantizer( params_.log2QuantizerSizeY_ );
      if ( afps.getRaw3dOffsetBitCountExplicitModeFlag() ) {
        ath.setRaw3dOffsetAxisBitCountMinus1( 0 );  //Note. need to be an encoder parameter
      } else {
        ath.setRaw3dOffsetAxisBitCountMinus1( params_.geometry3dCoordinatesBitdepth_ +
                                           ( params_.additionalProjectionPlaneMode_ > 0 ) -
                                           params_.geometryNominal2dBitdepth_ - 1 );
      }
      ath.setNumRefIdxActiveOverrideFlag( false );
      ath.setRefAtlasFrameListSpsFlag( true );
      ath.setRefAtlasFrameListIdx( 0 );
      PCCFrameContext& tile = context[i].getTile( ti );
      createPatchFrameDataStructure( context, tile, atl, i, ti );
      tile.setAtlIndex( context.getAtlasTileLayerList().size() - 1 ); //ajt::check setAtlIndex?
    }  // tileIdx
  }    // frameCount


  if ( params_.decodedAtlasInformationHash_ > 0 ) {
    context.allocateSeiHash( frameCount );
    assert( context.getSeiHash().size() == frameCount );
    for ( size_t fi = 0; fi < frameCount; fi++ ) {
      createHashInformation( context, fi, params_.decodedAtlasInformationHash_ - 1 );
    }
  }


  if ( params_.flagGeometrySmoothing_ ) {
    if ( params_.gridSmoothing_ ) {
      auto& sei = static_cast<SEIGeometrySmoothing&>( context.addSeiPrefix( GEOMETRY_SMOOTHING, true ) );
      sei.setPersistenceFlag( true );
      sei.setResetFlag( true );
      sei.setInstancesUpdated( true );
      sei.allocate();
      for ( size_t i = 0; i < sei.getInstancesUpdated(); i++ ) {
        size_t k = i;
        sei.setInstanceIndex( i, k );
        sei.setInstanceCancelFlag( k, false );
        sei.setMethodType( k, 1 );
        sei.setGridSizeMinus2( k, params_.gridSize_ - 2 );
        sei.setThreshold( k, params_.thresholdSmoothing_ );
      }
    }
    if ( params_.pbfEnableFlag_ ) {
      auto& sei = static_cast<SEIOccupancySynthesis&>( context.addSeiPrefix( OCCUPANCY_SYNTHESIS, true ) );
      sei.setPersistenceFlag( true );
      sei.setResetFlag( true );
      sei.setInstancesUpdated( true );
      sei.allocate();
      for ( size_t i = 0; i < sei.getInstancesUpdated(); i++ ) {
        size_t k = i;
        sei.setInstanceIndex( i, k );
        sei.setInstanceCancelFlag( k, false );
        sei.setMethodType( k, 1 );
        sei.setPbfLog2ThresholdMinus1( k, params_.pbfLog2Threshold_ - 1 );
        sei.setPbfPassesCountMinus1( k, params_.pbfPassesCount_ - 1 );
        sei.setPbfFilterSizeMinus1( k, params_.pbfFilterSize_ - 1 );
      }
    }
  }
  if ( params_.flagColorSmoothing_ ) {
    auto& sei = static_cast<SEIAttributeSmoothing&>( context.addSeiPrefix( ATTRIBUTE_SMOOTHING, true ) );
    if ( params_.flagColorSmoothing_ ) {
      sei.setPersistenceFlag( true );
      sei.setResetFlag( true );
      sei.setNumAttributesUpdated( 1 );
      sei.allocate();
      for ( size_t j = 0; j < sei.getNumAttributesUpdated(); j++ ) {
        size_t k = j;
        sei.setAttributeIdx( j, k );
        sei.setInstancesUpdated( k, 1 );
        sei.setAttributeSmoothingCancelFlag( k, false );
        sei.allocate( k + 1, sei.getInstancesUpdated( k ) + 1 );
        for ( size_t i = 0; i < sei.getInstancesUpdated( k ); i++ ) {
          size_t m = i;
          sei.setInstanceIndex( k, i, m );
          sei.setInstanceCancelFlag( k, m, false );
          sei.setMethodType( k, m, 1 );
          if ( sei.getMethodType( k, m ) == 1 ) {
            sei.setGridSizeMinus2( k, m, params_.cgridSize_ - 2 );
            sei.setThreshold( k, m, params_.thresholdColorSmoothing_ );
            sei.setThresholdVariation( k, m, params_.thresholdColorVariation_ );
            sei.setThresholdDifference( k, m, params_.thresholdColorDifference_ );
          }
        }
      }
    }
  }
#if 1
  printf( "createPatchFrameDataStructure done\n" );
#endif
}

void PCCEncoder::createPatchFrameDataStructure( PCCContext&         context,
                                                PCCFrameContext&    tile,
                                                AtlasTileLayerRbsp& atglu,
                                                size_t              frameIndex,
                                                size_t              tileIndex ) {
  TRACE_PATCH( "createPatchFrameDataStructure Tile %zu \n", tile.getFrameIndex() );
  auto&        patches            = tile.getPatches();
  auto&        pcmPatches         = tile.getRawPointsPatches();
  auto&        sps                = context.getVps();
  auto&        ath                = atglu.getHeader();
  auto&        atgdu              = atglu.getDataUnit();
  auto&        afps               = context.getAtlasFrameParameterSet( ath.getAtlasFrameParameterSetId() );
  size_t       aspsId             = afps.getAtlasSequenceParameterSetId();
  auto&        asps               = context.getAtlasSequenceParameterSet( aspsId );
  const int64_t minLevel           = pow( 2., ath.getPosMinDQuantizer() );
  size_t       atlasIndex         = context.getAtlasIndex();
  auto         geometryBitDepth2D = asps.getGeometry2dBitdepthMinus1() + 1;
  auto         geometryBitDepth3D = asps.getGeometry3dBitdepthMinus1() + 1;
  int64_t      prevSizeU0         = 0;
  int64_t      prevSizeV0         = 0;
  int64_t      predIndex          = 0;

  atglu.setAtlasFrmOrderCntVal( tile.getAtlasFrmOrderCntVal() );
  ath.setId( tileIndex );

  ath.setAtlasFrmOrderCntLsb( tile.getAtlasFrmOrderCntLsb() );
  ath.setType( I_TILE );    // P_TILE = 0, I_TILE, SKIP_TILE
  if ( frameIndex != 0 ) {  // && sps.getPatchInterPredictionEnabledFlag() ){
    bool interPredPresent = false;
    for ( auto& patch : patches ) {
      interPredPresent |= ( patch.getBestMatchIdx() != InvalidPatchIndex );
      if ( interPredPresent ) { break; }
    }
    if ( interPredPresent ) { ath.setType( P_TILE ); }
  }
  if ( ath.getType() == I_TILE ) {
    for ( auto& patch : patches ) { patch.setBestMatchIdx( InvalidPatchIndex ); }
    tile.setNumMatchedPatches( 0 );
  }

  if ( ath.getType() != I_TILE ) {
    tile.constructAtghRefListStruct( context, ath );
    TRACE_PATCH( "\tframe[%zu]\tRefAfocList:", frameIndex );
    for ( size_t i = 0; i < tile.getRefAfocListSize(); i++ ) { TRACE_CODEC( "\t%zu", tile.getRefAfoc( i ) ); }
    TRACE_PATCH( "\n" );
  }
  TRACE_PATCH( "Patches size                      = %zu \n", patches.size() );
  TRACE_PATCH( "non-regular Patches(raw, eom)     = %zu, %zu \n", tile.getRawPointsPatches().size(),
               tile.getEomPatches().size() );
  TRACE_PATCH( "Tile Type                         = %zu (0.P_TILE 1.I_TILE 2.SKIP_TILE)\n", (size_t)ath.getType() );

// all patches
#ifdef CODEC_TRACE
  size_t totalPatchCount = patches.size() + tile.getRawPointsPatches().size() + tile.getEomPatches().size();
#endif
  int32_t quantizerSizeX = 1 << params_.log2QuantizerSizeX_;
  int32_t quantizerSizeY = 1 << params_.log2QuantizerSizeY_;
  for ( size_t patchIndex = 0; patchIndex < patches.size(); patchIndex++ ) {
    const auto& patch = patches[patchIndex];
    if ( patch.getBestMatchIdx() != InvalidPatchIndex ) {
      // INTER patches
      size_t      refPOC   = (size_t)tile.getRefAfoc( patch.getRefAtlasFrameIndex() );
      const auto& refPatch = context.getFrame( refPOC ).getTile( tileIndex ).getPatches()[patch.getBestMatchIdx()];
      auto&       pid      = atgdu.addPatchInformationData( static_cast<uint8_t>( P_INTER ) );
      TRACE_PATCH( "patch %zu / %zu: Inter \n", patchIndex, totalPatchCount );
      auto& ipdu = pid.getInterPatchDataUnit();
      ipdu.setRefIndex( patch.getRefAtlasFrameIndex() );
      ipdu.setRefPatchIndex( static_cast<int64_t>( patch.getBestMatchIdx() ) - predIndex );
      ipdu.set2dPosX( patch.getU0() - refPatch.getU0() );
      ipdu.set2dPosY( patch.getV0() - refPatch.getV0() );
      if ( asps.getPatchSizeQuantizerPresentFlag() ) {
        int32_t deltaSizeX =
            patch.getPatchSize2DXInPixel() / quantizerSizeX - refPatch.getPatchSize2DXInPixel() / quantizerSizeX;
        int32_t deltaSizeY =
            patch.getPatchSize2DYInPixel() / quantizerSizeY - refPatch.getPatchSize2DYInPixel() / quantizerSizeY;
        ipdu.set2dDeltaSizeX( deltaSizeX );
        ipdu.set2dDeltaSizeY( deltaSizeY );
      } else {
        ipdu.set2dDeltaSizeX( patch.getSizeU0() - refPatch.getSizeU0() );
        ipdu.set2dDeltaSizeY( patch.getSizeV0() - refPatch.getSizeV0() );
      }
      ipdu.set3dOffsetU( patch.getU1() - refPatch.getU1() );
      ipdu.set3dOffsetV( patch.getV1() - refPatch.getV1() );

      const size_t max3DCoordinate =  geometryBitDepth3D;
      if ( patch.getProjectionMode() == 0 ) {
        ipdu.set3dOffsetD( ( patch.getD1() / minLevel ) - ( refPatch.getD1() / minLevel ) );
      } else {
        if ( static_cast<int>( asps.getExtendedProjectionEnabledFlag() ) == 0 ) {
          ipdu.set3dOffsetD( ( max3DCoordinate - patch.getD1() ) / minLevel -
                             ( max3DCoordinate - refPatch.getD1() ) / minLevel );
        } else {
          ipdu.set3dOffsetD( ( ( max3DCoordinate << 1 ) - patch.getD1() ) / minLevel -
                             ( ( max3DCoordinate << 1 ) - refPatch.getD1() ) / minLevel );
        }
      }
      int64_t diffDD  = (int64_t)patch.getSizeDPixel()-(int64_t)refPatch.getSizeD();
      int64_t quantDD = diffDD ==0? 0 : (diffDD +1)/ (int64_t)minLevel;
      ipdu.set3dRangeD( quantDD );
      int64_t delta_DD = quantDD==0? 0: (quantDD*minLevel-1); //(int64_t)refPatch.getSizeD() + delta_DD;
      patches[patchIndex].getSizeD() = (size_t) std::min( (int64_t) std::max((int64_t)(refPatch.getSizeD() + delta_DD), (int64_t)0), (int64_t)(1<<geometryBitDepth2D)-1);

      TRACE_PATCH(
          "\tIPDU: refAtlasFrame= %d refPatchIdx = %d pos2DXY = %ld %ld pos3DXYZW = %ld %ld %ld %ld size2D = %ld %ld "
          "\n",
          ipdu.getRefIndex(), ipdu.getRefPatchIndex(), ipdu.get2dPosX(), ipdu.get2dPosY(), ipdu.get3dOffsetU(),
          ipdu.get3dOffsetV(), ipdu.get3dOffsetD(), ipdu.get3dRangeD(), ipdu.get2dDeltaSizeX(),
          ipdu.get2dDeltaSizeY() );
      TRACE_PATCH(
          "\trefPatch: refIndex = %zu, refFrame = %zu, Idx = %zu/%zu UV0 = %zu %zu  UV1 = %zu %zu Size = %zu %zu %zu "
          " Lod = %u,%u\n",
          patch.getRefAtlasFrameIndex(), refPOC, patch.getBestMatchIdx(),
          context.getFrame( refPOC ).getTile( tileIndex ).getPatches().size(), refPatch.getU0(), refPatch.getV0(),
          refPatch.getU1(), refPatch.getV1(), refPatch.getSizeU0(), refPatch.getSizeV0(), refPatch.getSizeD(),
          refPatch.getLodScaleX(), refPatch.getLodScaleY() );

      if ( asps.getPLREnabledFlag() ) {
        setPLRData( tile, patch, ipdu.getPLRData(), size_t( 1 ) << asps.getLog2PatchPackingBlockSize(), patchIndex );
      }
      prevSizeU0 = asps.getPatchSizeQuantizerPresentFlag() ? patch.getPatchSize2DXInPixel()
                                                           : patch.getSizeU0();  // prevPatchSize2DXInPixel
      prevSizeV0 = asps.getPatchSizeQuantizerPresentFlag() ? patch.getPatchSize2DYInPixel() : patch.getSizeV0();
      predIndex += ipdu.getRefPatchIndex() + 1;

      TRACE_PATCH(
          "\tpatch(Inter) %zu: UV0 %4zu %4zu UV1 %4zu %4zu D1=%4zu S=%4zu %4zu %4zu from DeltaSize = %4ld %4ld P=%zu "
          "O=%zu A=%u%u%u Lod = %zu,%zu \n",
          patchIndex, patch.getU0(), patch.getV0(), patch.getU1(), patch.getV1(), patch.getD1(), patch.getSizeU0(),
          patch.getSizeV0(), patch.getSizeD(), ipdu.get2dDeltaSizeX(), ipdu.get2dDeltaSizeY(),
          patch.getProjectionMode(), patch.getPatchOrientation(), patch.getNormalAxis(), patch.getTangentAxis(),
          patch.getBitangentAxis(), patch.getLodScaleX(), patch.getLodScaleY() );
    } else {
      // INTRA patches
      uint8_t patchType = static_cast<uint8_t>( ( ath.getType() == I_TILE ) ? I_INTRA : P_INTRA );
      auto&   pid       = atgdu.addPatchInformationData( patchType );
      TRACE_PATCH( "patch %zu / %zu: Intra \n", patchIndex, totalPatchCount );
      auto& pdu = pid.getPatchDataUnit();
      pdu.set2dPosX( patch.getU0() );
      pdu.set2dPosY( patch.getV0() );
      bool lodEnableFlag = ( patch.getLodScaleX() > 1 || patch.getLodScaleY() > 1 );
      // PatchLoDScaleX[ p ] = pdu_lod_enable_flag[ p ] ?
      // pdu_lod_scale_x_minus1[ p ] + 1: 1
      // PatchLoDScaleY[ p ] = pdu_lod_enable_flag[ p ] ? (pdu_lod_scale_y[ p
      // ] + (pdu_lod_scale_x_minus1[ p ] > 0) ?
      // 1 : 2) : 1
      if ( afps.getLodModeEnableFlag() ) {
        pdu.setLodEnableFlag( lodEnableFlag );
        if ( lodEnableFlag ) {
          pdu.setLodScaleXMinus1( patch.getLodScaleX() - 1 );
          pdu.setLodScaleYIdc( patch.getLodScaleY() - ( patch.getLodScaleX() > 1 ? 1 : 2 ) );
        }
      } else {
        pdu.setLodEnableFlag( false );
        pdu.setLodScaleXMinus1( 0 );
        pdu.setLodScaleYIdc( 0 );
      }
      pdu.set3dOffsetU( patch.getU1() );
      pdu.set3dOffsetV( patch.getV1() );
      pdu.setProjectionId( patch.getViewId() );
      if ( asps.getPatchSizeQuantizerPresentFlag() ) {
        pdu.set2dSizeXMinus1( ( patch.getPatchSize2DXInPixel() - 1 ) / quantizerSizeX );
        pdu.set2dSizeYMinus1( ( patch.getPatchSize2DYInPixel() - 1 ) / quantizerSizeY );
      } else {
        pdu.set2dSizeXMinus1( patch.getSizeU0() - 1 );
        pdu.set2dSizeYMinus1( patch.getSizeV0() - 1 );
      }
      pdu.setOrientationIndex( patch.getPatchOrientation() );
      const size_t max3DCoordinate = size_t( 1 ) << ( geometryBitDepth3D );
      if ( patch.getProjectionMode() == 0 ) {
        pdu.set3dOffsetD( patch.getD1() / minLevel );
      } else {
        pdu.set3dOffsetD( ( max3DCoordinate - patch.getD1() ) / minLevel );
      }
      // Note: quantDD cannot cover up to the maximum depth by this equation. (e.g.getSizeD=255)
      size_t quantDD = patch.getSizeD() == 0 ? 0 : ( ( patch.getSizeD() + 1 ) / minLevel );
      pdu.set3dRangeD( quantDD );

      TRACE_PATCH(
          "patch(Intra) %zu: UV0 %4zu %4zu UV1 %4zu %4zu D1=%4zu S=%4zu %4zu %4zu(%4zu) P=%zu O=%zu A=%u%u%u Lod "
          "=(%zu) %zu,%zu 45=%d ProjId=%4zu Axis=%zu \n",
          patchIndex, patch.getU0(), patch.getV0(), patch.getU1(), patch.getV1(), patch.getD1(), patch.getSizeU0(),
          patch.getSizeV0(), patch.getSizeD(), pdu.get3dRangeD(), patch.getProjectionMode(),
          patch.getPatchOrientation(), patch.getNormalAxis(), patch.getTangentAxis(), patch.getBitangentAxis(),
          (size_t)lodEnableFlag, patch.getLodScaleX(), patch.getLodScaleY(), asps.getExtendedProjectionEnabledFlag(),
          pdu.getProjectionId(), patch.getAxisOfAdditionalPlane() );

      if ( asps.getPLREnabledFlag() ) {
        setPLRData( tile, patch, pdu.getPLRData(), size_t( 1 ) << asps.getLog2PatchPackingBlockSize(), patchIndex );
      }
    }
  }

  if ( ( params_.losslessGeo_ || params_.lossyRawPointsPatch_ ) ) {
    size_t numberOfPcmPatches = tile.getNumberOfRawPointsPatches();
    for ( size_t mpsPatchIndex = 0; mpsPatchIndex < numberOfPcmPatches; ++mpsPatchIndex ) {
      auto&   rawPointsPatch = pcmPatches[mpsPatchIndex];
      uint8_t patchType      = static_cast<uint8_t>( ( ath.getType() == I_TILE ) ? I_RAW : P_RAW );
      auto&   pid            = atgdu.addPatchInformationData( patchType );
      auto&   rpdu           = pid.getRawPatchDataUnit();
      TRACE_PATCH( "patch %zu / %zu: raw \n", patches.size() + mpsPatchIndex, totalPatchCount );
      rpdu.setPatchInAuxiliaryVideoFlag( rawPointsPatch.isPatchInAuxVideo_ );
      rpdu.set2dPosX( rawPointsPatch.u0_ );
      rpdu.set2dPosY( rawPointsPatch.v0_ );
      rpdu.set2dSizeXMinus1( rawPointsPatch.sizeU0_ - 1 );
      rpdu.set2dSizeYMinus1( rawPointsPatch.sizeV0_ - 1 );
      if ( afps.getRaw3dOffsetBitCountExplicitModeFlag() ) {
        rpdu.set3dOffsetU( rawPointsPatch.u1_ );
        rpdu.set3dOffsetV( rawPointsPatch.v1_ );
        rpdu.set3dOffsetD( rawPointsPatch.d1_ );
      } else {
        const size_t pcmU1V1D1Level = size_t( 1 ) << ( asps.getGeometry2dBitdepthMinus1() + 1 );
        rpdu.set3dOffsetU( rawPointsPatch.u1_ / pcmU1V1D1Level );
        rpdu.set3dOffsetV( rawPointsPatch.v1_ / pcmU1V1D1Level );
        rpdu.set3dOffsetD( rawPointsPatch.d1_ / pcmU1V1D1Level );
      }
      rpdu.setPatchInAuxiliaryVideoFlag( sps.getAuxiliaryVideoPresentFlag( 0 ) );
      rpdu.setRawPointsMinus1( uint32_t( rawPointsPatch.getNumberOfRawPoints() - 1 ) );
      TRACE_PATCH( "Raw :UV = %zu %zu  size = %zu %zu  uvd1 = %zu %zu %zu numPoints = %zu ocmRes = %zu \n",
                   rawPointsPatch.u0_, rawPointsPatch.v0_, rawPointsPatch.sizeU0_, rawPointsPatch.sizeV0_,
                   rawPointsPatch.u1_, rawPointsPatch.v1_, rawPointsPatch.d1_, rawPointsPatch.getNumberOfRawPoints(),
                   rawPointsPatch.occupancyResolution_ );
    }
  }
  if ( params_.enhancedOccupancyMapCode_ ) {
    size_t numberOfEomPatches = tile.getEomPatches().size();
    for ( size_t eomPatchIndex = 0; eomPatchIndex < numberOfEomPatches; ++eomPatchIndex ) {
      auto&   eomPatch  = tile.getEomPatches()[eomPatchIndex];
      uint8_t patchType = static_cast<uint8_t>( ( ath.getType() == I_TILE ) ? I_EOM : P_EOM );
      auto&   pid       = atgdu.addPatchInformationData( patchType );
      auto&   epdu      = pid.getEomPatchDataUnit();
      TRACE_PATCH( "patch %zu / %zu: EOM \n", patches.size() + pcmPatches.size() + eomPatchIndex, totalPatchCount );
      epdu.setPatchInAuxiliaryVideoFlag( eomPatch.isPatchInAuxVideo_ );
      epdu.set2dPosX( eomPatch.u0_ );
      epdu.set2dPosY( eomPatch.v0_ );
      epdu.set2dSizeXMinus1( eomPatch.sizeU_ - 1 );
      epdu.set2dSizeYMinus1( eomPatch.sizeV_ - 1 );
      assert( !eomPatch.memberPatches.empty() );
      epdu.setPatchCountMinus1( uint32_t( eomPatch.memberPatches.size() - 1 ) );
      for ( size_t i = 0; i < eomPatch.memberPatches.size(); i++ ) {
        epdu.setAssociatedPatchesIdx( i, eomPatch.memberPatches[i] );
        epdu.setPoints( i, eomPatch.eomCountPerPatch[i] );
      }
      TRACE_PATCH( "EOM: U0V0 %zu,%zu\tSizeU0V0 %zu,%zu\tN= %zu,%zu\n", eomPatch.u0_, eomPatch.v0_, eomPatch.sizeU_,
                   eomPatch.sizeV_, eomPatch.memberPatches.size(), eomPatch.eomCount_ );
      for ( size_t i = 0; i < eomPatch.memberPatches.size(); i++ ) {
        TRACE_PATCH( "%zu, %zu\n", eomPatch.memberPatches[i], eomPatch.eomCountPerPatch[i] );
      }
    }
  }
  TRACE_PATCH( "patch %zu / %zu: end \n", patches.size(), patches.size() );
  uint8_t patchType = static_cast<uint8_t>( ( ath.getType() == I_TILE ) ? I_END : P_END );
  atgdu.addPatchInformationData( patchType );
}

void PCCEncoder::createHashInformation( PCCContext& context, int frameIndex, size_t hashType ) {
  TRACE_PATCH( "createHashInformation Frame %zu \n", frameIndex );
  size_t                                         hashIndex = frameIndex;
  std::vector<PatchParams>                       atlasPatchParams;
  std::vector<std::vector<PatchParams>>          tilePatchParams;
  std::vector<std::vector<std::vector<int64_t>>> tileB2PPatchParams;
  std::vector<std::vector<int64_t>>              atlasB2PPatchParams;

  auto& sei = context.getSeiHash( hashIndex );
  sei.setDecodedHighLevelHashPresentFlag( frameIndex >= 0 );
  sei.setDecodedAtlasHashPresentFlag( frameIndex >= 0 );
  sei.setDecodedAtlasB2pHashPresentFlag( true );
  sei.setDecodedAtlasTilesHashPresentFlag( context[frameIndex].getNumTilesInAtlasFrame() >= 0 );
  sei.setDecodedAtlasTilesB2pHashPresentFlag( context[frameIndex].getNumTilesInAtlasFrame() >= 0 );
  sei.setCancelFlag( false );
  sei.setHashType( hashType );
  bool seiHashCancelFlag = sei.getCancelFlag();

  // parameter sets
  if ( sei.getDecodedHighLevelHashPresentFlag() && !seiHashCancelFlag ) {
    size_t               atlIdx     = context[frameIndex].getTile( 0 ).getAtlIndex();
    auto&                tileHeader = context.getAtlasTileLayerList()[atlIdx].getHeader();
    size_t               afpsIndex  = tileHeader.getAtlasFrameParameterSetId();
    size_t               aspsIndex  = context.getAtlasFrameParameterSet( afpsIndex ).getAtlasFrameParameterSetId();
    auto&                asps       = context.getAtlasSequenceParameterSet( aspsIndex );
    auto&                afps       = context.getAtlasFrameParameterSet( afpsIndex );
    std::vector<uint8_t> highLevelAtlasData;
    aspsCommonByteString( highLevelAtlasData, asps );
    aspsApplicationByteString( highLevelAtlasData, asps, afps );
    afpsCommonByteString( highLevelAtlasData, context, afpsIndex, frameIndex );
    afpsApplicationByteString( highLevelAtlasData, asps, afps );
    printf( "**sei** HighLevel Hash\n" );
    if ( sei.getHashType() == 0 ) {
      std::vector<uint8_t> encMD5( 16 );
      encMD5 = context.computeMD5( highLevelAtlasData.data(), highLevelAtlasData.size() );
      printf( "\t**sei** (MD5): " );
      for ( auto& e : encMD5 ) printf( "%02x", e );
      printf( "\n" );
      for ( int j = 0; j < 16; j++ ) sei.setHighLevelMd5( j, encMD5[j] );
    } else if ( sei.getHashType() == 1 ) {
      uint16_t crc = context.computeCRC( highLevelAtlasData.data(), highLevelAtlasData.size() );
      printf( "\t**sei** (CRC): %04x\n", crc );
      sei.setHighLevelCrc( crc );
    } else if ( sei.getHashType() == 2 ) {
      uint32_t checkSum = context.computeCheckSum( highLevelAtlasData.data(), highLevelAtlasData.size() );
      printf( "\t**sei** (CheckSum): %08x\n", checkSum );
      sei.setHighLevelCheckSum( checkSum );
    }
    highLevelAtlasData.clear();
  }

  if ( !seiHashCancelFlag && ( sei.getDecodedAtlasTilesHashPresentFlag() || sei.getDecodedAtlasHashPresentFlag() ) ) {
    size_t numTilesInPatchFrame = context[frameIndex].getNumTilesInAtlasFrame();
    if ( sei.getDecodedAtlasTilesHashPresentFlag() ) tilePatchParams.resize( numTilesInPatchFrame );
    for ( size_t tileIdx = 0; tileIdx < numTilesInPatchFrame; tileIdx++ ) {
      auto& tile = context[frameIndex].getTile( tileIdx );
      getHashPatchParams( context, frameIndex, tileIdx, tile.getAtlIndex(), tilePatchParams, atlasPatchParams );
    }
  }
  if ( !seiHashCancelFlag &&
       ( sei.getDecodedAtlasB2pHashPresentFlag() || sei.getDecodedAtlasTilesB2pHashPresentFlag() ) ) {
    getB2PHashPatchParams( context, frameIndex, tileB2PPatchParams, atlasB2PPatchParams );
  }

  // per frame
  if ( sei.getDecodedAtlasHashPresentFlag() && !seiHashCancelFlag ) {
    std::vector<uint8_t> atlasData;
    size_t               patchCount = atlasPatchParams.size();
    for ( size_t atlasPatchIdx = 0; atlasPatchIdx < patchCount; atlasPatchIdx++ ) {
      atlasPatchCommonByteString( atlasData, atlasPatchIdx, atlasPatchParams );
      atlasPatchApplicationByteString( atlasData, atlasPatchIdx, atlasPatchParams );
    }

    size_t atlIdx     = context[frameIndex].getTile( 0 ).getAtlIndex();
    auto&  tileHeader = context.getAtlasTileLayerList()[atlIdx].getHeader();
    size_t afpsIndex  = tileHeader.getAtlasFrameParameterSetId();
    size_t aspsIndex  = context.getAtlasFrameParameterSet( afpsIndex ).getAtlasFrameParameterSetId();
    auto& afps = context.getAtlasFrameParameterSet( afpsIndex);
    auto& asps = context.getAtlasSequenceParameterSet( aspsIndex );
     auto& vps = context.getVps();
     auto& afc = context.getFrames()[frameIndex];
     auto& tfc        = afc.getTitleFrameContext();
     TRACE_ATLAS( "Atlas Frame Index = %d,\n", frameIndex);
     TRACE_ATLAS("AtlasFrameOrderCntVal = %d,  AtlasFrameWidthMax =  %d, AtlasFrameHeightMax = %d, AtlasID = %d, ASPSFrameSize = %d, VPSMapCount = %d, AttributeCount = %d, AttributeDimension = %d, NumTilesAtlasFrame = %d, AtlasTotalNumProjPatches = %d, " " AtlasTotalNumRawPatches = %d, AtlasTotalNumEOMPatches = %d, ",
                 tfc.getAtlasFrmOrderCntVal(), asps.getFrameWidth(), asps.getFrameHeight(), vps.getAtlasId( 0 ),
                 asps.getFrameWidth() * asps.getFrameHeight(), vps.getMapCountMinus1(0) + 1,
                 vps.getAttributeInformation(0).getAttributeCount(),
                 vps.getAttributeInformation(0).getAttributeDimensionMinus1(0) + 1, afc.getNumTilesInAtlasFrame(),
                 tfc.getPatches().size(), tfc.getNumberOfRawPointsPatches(), tfc.getEomPatches().size() );
     TRACE_FRAME( "Atlas Frame Index = %d, \n", frameIndex );
     TRACE_FRAME( "PointCloudFrameOrderCntVal = %d, NumProjPoints = %zu, NumRAWPoints = %zu, NumEOMPoints = %zu,\n",
                  tfc.getAtlasFrmOrderCntVal(), tfc.getTotalNumberOfRegularPoints(), tfc.getTotalNumberOfRawPoints(),
                  tfc.getTotalNumberOfEOMPoints() );
                       
    printf( "**sei** AtlasPatchHash: frame(%d) (#patches %zu)\n", frameIndex, patchCount );
    if ( sei.getHashType() == 0 ) {
      std::vector<uint8_t> md5Digest( 16 );
      md5Digest = context.computeMD5( atlasData.data(), atlasData.size() );
      printf( "\t**sei** (MD5): " );
      for ( auto& e : md5Digest ) printf( "%02x", e );
      printf( "\n" );
      TRACE_ATLAS( " Atlas MD5 = " );
      for ( int j = 0; j < 16; j++ ) { 
          sei.setAtlasMd5( j, md5Digest[j] );
          TRACE_ATLAS( "%02x", md5Digest[j] );
      }
      TRACE_ATLAS( "," );
    } else if ( sei.getHashType() == 1 ) {
      uint16_t crc = context.computeCRC( atlasData.data(), atlasData.size() );
      printf( "\t**sei** (CRC): % 04x ", crc );
      sei.setAtlasCrc( crc );
    } else if ( sei.getHashType() == 2 ) {
      uint32_t checkSum = context.computeCheckSum( atlasData.data(), atlasData.size() );
      printf( "\t**sei** (checkSum): % 08x ", checkSum );
      sei.setAtlasCheckSum( checkSum );
    }
    atlasData.clear();
  }

  if ( sei.getDecodedAtlasB2pHashPresentFlag() && !seiHashCancelFlag ) {
    std::vector<uint8_t> atlasB2PData;
    atlasBlockToPatchByteString( atlasB2PData, atlasB2PPatchParams );
    printf( "**sei** AtlasBlockToPatchHash: frame(%d) \n", frameIndex );
    if ( sei.getHashType() == 0 ) {
      std::vector<uint8_t> md5Digest( 16 );
      md5Digest = context.computeMD5( atlasB2PData.data(), atlasB2PData.size() );
      printf( "\t**sei** (MD5): " );
      for ( auto& e : md5Digest ) printf( "%02x", e );
      printf( "\n" );
      TRACE_ATLAS( " Atlas B2P MD5 = " );
      for ( int j = 0; j < 16; j++ ) { 
          sei.setAtlasB2pMd5( j, md5Digest[j] ); 
          TRACE_ATLAS( "%02x", md5Digest[j] );
      }
      TRACE_ATLAS( "," );
    } else if ( sei.getHashType() == 1 ) {
      uint16_t crc = context.computeCRC( atlasB2PData.data(), atlasB2PData.size() );
      printf( "\t**sei** (CRC): % 04x ", crc );
      sei.setAtlasB2pCrc( crc );
    } else if ( sei.getHashType() == 2 ) {
      uint32_t checkSum = context.computeCheckSum( atlasB2PData.data(), atlasB2PData.size() );
      printf( "\t**sei** (checkSum): % 08x ", checkSum );
      sei.setAtlasB2pCheckSum( checkSum );
    }
    atlasB2PData.clear();
    TRACE_ATLAS( "\n" );
  }

  // for tiles
  if ( ( sei.getDecodedAtlasTilesHashPresentFlag() || sei.getDecodedAtlasTilesB2pHashPresentFlag() ) &&
       !seiHashCancelFlag ) {
    sei.allocateAtlasTilesHash( context[frameIndex].getNumTilesInAtlasFrame() );
    sei.setNumTilesMinus1( context[frameIndex].getNumTilesInAtlasFrame() - 1 );
    TRACE_TILE( "Atlas Frame Index = %d, \n", frameIndex );
    for ( size_t tileIdx = 0; tileIdx < context[frameIndex].getNumTilesInAtlasFrame(); tileIdx++ ) {
      auto&  tile         = context[frameIndex].getTile( tileIdx );
      size_t atlIdx       = tile.getAtlIndex();
      auto&  tileHeader   = context.getAtlasTileLayerList()[atlIdx].getHeader();
      auto&  tileDataUnit = context.getAtlasTileLayerList()[atlIdx].getDataUnit();
      size_t patchCount   = tileDataUnit.getPatchCount() - 1;  // not the last I_END or P_END
      size_t tileId       = tileHeader.getId();
      auto&  afps         = context.getAtlasFrameParameterSet( tileHeader.getAtlasFrameParameterSetId());
      auto&   afti              = afps.getAtlasFrameTileInformation();
      size_t topLeftColumn     = afti.getTopLeftPartitionIdx( tileIdx ) % ( afti.getNumPartitionColumnsMinus1() + 1 );
      size_t topLeftRow        = afti.getTopLeftPartitionIdx( tileIdx ) / ( afti.getNumPartitionColumnsMinus1() + 1 );
      size_t  tileOffsetX       = context[frameIndex].getPartitionPosX(topLeftColumn);
      size_t  tileOffsetY       = context[frameIndex].getPartitionPosY(topLeftRow);
      TRACE_TILE( "TileID = %d, AtlasFrameOrderCntVal = %d, TileType = %d, TileOffsetX = %d, TileOffsetY = %d, TileWidth = %d, TileHeight = %d, ", 
                  tileId , tile.getAtlasFrmOrderCntVal(), tileHeader.getType(), tileOffsetX, tileOffsetY,
                  tile.getWidth(), tile.getHeight() );
      sei.setTileId( tileIdx, tileId );
      if ( tileIdx == 0 ) {
        auto& tileInfo = context.getAtlasFrameParameterSet( tileHeader.getAtlasFrameParameterSetId() )
                             .getAtlasFrameTileInformation();
        /*ajt:: bitCount - 1 could cause infinite loop when bitCount = 0,proposed solution comes with an extra bit cost
        uint8_t bitCount = tileInfo.getSignalledTileIdFlag()
                               ? ( tileInfo.getSignalledTileIdLengthMinus1() + 1 )
                               : ( tileInfo.getNumTilesInAtlasFrameMinus1() == 0
                                       ? 0
                                       : ( ceilLog2( tileInfo.getNumTilesInAtlasFrameMinus1() + 1 ) ) );
        sei.setTileIdLenMinus1( bitCount - 1 );*/
        uint8_t bitCount = tileInfo.getSignalledTileIdFlag()
                               ? ( tileInfo.getSignalledTileIdLengthMinus1() + 1 )
                               : ( ceilLog2( tileInfo.getNumTilesInAtlasFrameMinus1() + 1 ) );
        sei.setTileIdLenMinus1( tileInfo.getNumTilesInAtlasFrameMinus1() == 0 ? 0 : ( bitCount - 1 ) );
      }
      if ( sei.getDecodedAtlasTilesHashPresentFlag() ) {
        std::vector<uint8_t> atlasTileData;
        for ( size_t patchIdx = 0; patchIdx < patchCount; patchIdx++ ) {
          tilePatchCommonByteString( atlasTileData, tileId, patchIdx, tilePatchParams );
          tilePatchApplicationByteString( atlasTileData, tileId, patchIdx, tilePatchParams );
        }
        printf( "**sei** TilesPatchHash: frame(%d), tile(%zu, tileId %zu)\n", frameIndex, tileIdx, tileId );
        if ( sei.getHashType() == 0 ) {
          std::vector<uint8_t> md5Digest( 16 );
          md5Digest = context.computeMD5( atlasTileData.data(), atlasTileData.size() );
          printf( "\t**sei** (MD5): " );
          for ( auto& e : md5Digest ) printf( "%02x", e );
          printf( "\n" );
          TRACE_TILE( " Tile MD5 = ");
          for ( int j = 0; j < 16; j++ ) { 
              sei.setAtlasTilesMd5( tileId, j, md5Digest[j] );
              TRACE_TILE( "%02x", md5Digest[j] );
          }
          TRACE_TILE( "," );
        } else if ( sei.getHashType() == 1 ) {
          uint16_t crc = context.computeCRC( atlasTileData.data(), atlasTileData.size() );
          printf( "\t**sei** (crc): % 02x ", crc );
          sei.setAtlasTilesCrc( tileId, crc );
        } else if ( sei.getHashType() == 2 ) {
          uint32_t checkSum = context.computeCheckSum( atlasTileData.data(), atlasTileData.size() );
          printf( "\t**sei** (checkSum): % 08x ", checkSum );
          sei.setAtlasTilesCheckSum( tileId, checkSum );
        }
        atlasTileData.clear();
      }
      if ( sei.getDecodedAtlasTilesB2pHashPresentFlag() ) {
        std::vector<uint8_t> tileB2PData;
        tileBlockToPatchByteString( tileB2PData, tileId, tileB2PPatchParams );
        printf( "**sei** TilesB2pPatchHash: frame(%d), tileIdx(%zu)\n", frameIndex, tileIdx );
        if ( sei.getHashType() == 0 ) {
          std::vector<uint8_t> md5Digest( 16 );
          md5Digest = context.computeMD5( tileB2PData.data(), tileB2PData.size() );
          printf( "\t**sei** (MD5): " );
          for ( auto& e : md5Digest ) printf( "%02x", e );
          printf( "\n" );
          TRACE_TILE( " Tile B2P MD5 = ");
          for ( int j = 0; j < 16; j++ ) {
            sei.setAtlasTilesB2pMd5( tileId, j, md5Digest[j] );
            TRACE_TILE( "%02x", md5Digest[j] );
          }
          TRACE_TILE( "," );
        } else if ( sei.getHashType() == 1 ) {
          uint16_t crc = context.computeCRC( tileB2PData.data(), tileB2PData.size() );
          printf( "\t**sei** (CRC): % 04x ", crc );
          sei.setAtlasTilesB2pCrc( tileId, crc );
        } else if ( sei.getHashType() == 2 ) {
          uint32_t checkSum = context.computeCheckSum( tileB2PData.data(), tileB2PData.size() );
          printf( "\t**sei** (checkSum): % 08x ", checkSum );
          sei.setAtlasTilesB2pCheckSum( tileId, checkSum );
        }
        tileB2PData.clear();
      }
      TRACE_TILE( "\n" );
    }
  }
  context.addSeiHashToSeiSuffix( hashIndex );
  if ( atlasPatchParams.size() != 0 ) atlasPatchParams.clear();
  if ( tilePatchParams.size() != 0 ) {
    for ( size_t ti = 0; ti < tilePatchParams.size(); ti++ )
      if ( tilePatchParams[ti].size() != 0 ) tilePatchParams[ti].clear();
  }
  tilePatchParams.clear();
  for ( auto& e : atlasB2PPatchParams ) e.clear();
  atlasB2PPatchParams.clear();
  for ( auto& e : tileB2PPatchParams ) {
    for ( auto d : e ) d.clear();
    e.clear();
  }
  tileB2PPatchParams.clear();
}
void PCCEncoder::segmentationPartiallyAddtinalProjectionPlane( const PCCPointSet3&                 source,
                                                               PCCFrameContext&                    frame,
                                                               const PCCPatchSegmenter3Parameters& segmenterParams,
                                                               size_t                              frameIndex,
                                                               float&                              distanceSrcRec ) {
  std::vector<PCCPatch> Orthogonal;
  std::vector<PCCPatch> Additional;
  size_t                axis  = 0;
  int                   min_x = ( 1U << segmenterParams.geometryBitDepth3D_ ) + 1;
  int                   max_x = -1;
  int                   min_y = min_x;
  int                   max_y = -1;
  int                   min_z = min_x;
  int                   max_z = -1;
  for ( size_t i = 0; i < source.getPointCount(); i++ ) {
    PCCPoint3D point = source[i];
    if ( min_x > point.x() ) { min_x = point.x(); }
    if ( min_y > point.y() ) { min_y = point.y(); }
    if ( min_z > point.z() ) { min_z = point.z(); }
    if ( max_x < point.x() ) { max_x = point.x(); }
    if ( max_y < point.y() ) { max_y = point.y(); }
    if ( max_z < point.z() ) { max_z = point.z(); }
  }
  int Id = 0;
  if ( max_x - min_x > max_y - min_y ) {
    Id = 1;
  } else {
    Id = 2;
  }
  if ( Id == 1 && max_z - min_z > max_x - min_x ) { Id = 3; }
  if ( Id == 2 && max_z - min_z > max_y - min_y ) { Id = 3; }

  // Id is 1:X, Id is 2:Y Id is 3:Z
  axis = Id;
  PCCPointSet3 partial;
  partial.clear();
  partial.addColors();

  double ratio = 1.0 - params_.partialAdditionalProjectionPlane_;
  for ( size_t i = 0; i < source.getPointCount(); i++ ) {
    PCCPoint3D point = source[i];
    PCCColor3B color = source.getColor( i );  // finally recolor color was used.

    if ( axis == 1 ) {
      if ( point.x() > min_x + ( max_x - min_x ) * ratio ) {
        PCCVector3D pos;
        pos.x() = point.x();
        pos.y() = point.y();
        pos.z() = point.z();
        partial.addPoint( pos, color );
      }
    }
    if ( axis == 2 ) {
      if ( point.y() > min_y + ( max_y - min_y ) * ratio ) {
        PCCVector3D pos;
        pos.x() = point.x();
        pos.y() = point.y();
        pos.z() = point.z();
        partial.addPoint( pos, color );
      }
    }
    if ( axis == 3 ) {
      if ( point.z() > min_z + ( max_z - min_z ) * ratio ) {
        PCCVector3D pos;
        pos.x() = point.x();
        pos.y() = point.y();
        pos.z() = point.z();
        partial.addPoint( pos, color );
      }
    }
  }

  {  // orthogonal 6 projection
    std::vector<PCCPointSet3>    Tmp;
    std::vector<PCCPointSet3>    PointCloudByPatchA;
    PCCPointSet3                 resampleKeepA;
    PCCPatchSegmenter3Parameters local   = segmenterParams;
    local.additionalProjectionPlaneMode_ = 0;
    PCCPatchSegmenter3 segmenter;
    Orthogonal.reserve( 256 );
    float distanceSrcRecA;
    segmenter.setNbThread( params_.nbThread_ );
    segmenter.compute( source, frame.getFrameIndex(), local, Orthogonal, frame.getSrcPointCloudByPatch(),
                       distanceSrcRecA );
    distanceSrcRec                  = distanceSrcRecA;
    frame.getSrcPointCloudByPatch() = Tmp;
  }

  if ( partial.getPointCount() != 0u ) {
    // additional projection
    std::vector<PCCPointSet3>    Tmp;
    std::vector<PCCPointSet3>    PointCloudByPatchA;
    PCCPointSet3                 resampleKeepA;
    PCCPatchSegmenter3Parameters local = segmenterParams;

    if ( axis == 1 ) { local.additionalProjectionPlaneMode_ = 2; }
    if ( axis == 2 ) { local.additionalProjectionPlaneMode_ = 1; }
    if ( axis == 3 ) { local.additionalProjectionPlaneMode_ = 3; }

    PCCPatchSegmenter3 segmenter;
    Additional.reserve( 256 );
    float distanceSrcRecA;
    segmenter.setNbThread( params_.nbThread_ );
    segmenter.compute( partial, frame.getFrameIndex(), local, Additional, frame.getSrcPointCloudByPatch(),
                       distanceSrcRecA );
    distanceSrcRec                  = distanceSrcRecA;
    frame.getSrcPointCloudByPatch() = Tmp;

    // remove
    const size_t        patchCount = Additional.size();
    int                 patchIndex;
    std::vector<size_t> remove;
    remove.clear();
    for ( patchIndex = patchCount - 1; patchIndex > -1; --patchIndex ) {
      auto& patch = Additional[patchIndex];
      if ( patch.getAxisOfAdditionalPlane() == 0 ) { remove.push_back( patchIndex ); }
    }
    // erace
    for ( auto& itr : remove ) { Additional.erase( Additional.begin() + itr ); }
  }
  auto& patches = frame.getPatches();
  patches.reserve( Orthogonal.size() + Additional.size() );
  std::copy( Orthogonal.begin(), Orthogonal.end(), std::back_inserter( patches ) );
  std::copy( Additional.begin(), Additional.end(), std::back_inserter( patches ) );
}

// Morton
const uint32_t kMortonCode256Z[256] = {
    0x00000000, 0x00000001, 0x00000008, 0x00000009, 0x00000040, 0x00000041, 0x00000048, 0x00000049, 0x00000200,
    0x00000201, 0x00000208, 0x00000209, 0x00000240, 0x00000241, 0x00000248, 0x00000249, 0x00001000, 0x00001001,
    0x00001008, 0x00001009, 0x00001040, 0x00001041, 0x00001048, 0x00001049, 0x00001200, 0x00001201, 0x00001208,
    0x00001209, 0x00001240, 0x00001241, 0x00001248, 0x00001249, 0x00008000, 0x00008001, 0x00008008, 0x00008009,
    0x00008040, 0x00008041, 0x00008048, 0x00008049, 0x00008200, 0x00008201, 0x00008208, 0x00008209, 0x00008240,
    0x00008241, 0x00008248, 0x00008249, 0x00009000, 0x00009001, 0x00009008, 0x00009009, 0x00009040, 0x00009041,
    0x00009048, 0x00009049, 0x00009200, 0x00009201, 0x00009208, 0x00009209, 0x00009240, 0x00009241, 0x00009248,
    0x00009249, 0x00040000, 0x00040001, 0x00040008, 0x00040009, 0x00040040, 0x00040041, 0x00040048, 0x00040049,
    0x00040200, 0x00040201, 0x00040208, 0x00040209, 0x00040240, 0x00040241, 0x00040248, 0x00040249, 0x00041000,
    0x00041001, 0x00041008, 0x00041009, 0x00041040, 0x00041041, 0x00041048, 0x00041049, 0x00041200, 0x00041201,
    0x00041208, 0x00041209, 0x00041240, 0x00041241, 0x00041248, 0x00041249, 0x00048000, 0x00048001, 0x00048008,
    0x00048009, 0x00048040, 0x00048041, 0x00048048, 0x00048049, 0x00048200, 0x00048201, 0x00048208, 0x00048209,
    0x00048240, 0x00048241, 0x00048248, 0x00048249, 0x00049000, 0x00049001, 0x00049008, 0x00049009, 0x00049040,
    0x00049041, 0x00049048, 0x00049049, 0x00049200, 0x00049201, 0x00049208, 0x00049209, 0x00049240, 0x00049241,
    0x00049248, 0x00049249, 0x00200000, 0x00200001, 0x00200008, 0x00200009, 0x00200040, 0x00200041, 0x00200048,
    0x00200049, 0x00200200, 0x00200201, 0x00200208, 0x00200209, 0x00200240, 0x00200241, 0x00200248, 0x00200249,
    0x00201000, 0x00201001, 0x00201008, 0x00201009, 0x00201040, 0x00201041, 0x00201048, 0x00201049, 0x00201200,
    0x00201201, 0x00201208, 0x00201209, 0x00201240, 0x00201241, 0x00201248, 0x00201249, 0x00208000, 0x00208001,
    0x00208008, 0x00208009, 0x00208040, 0x00208041, 0x00208048, 0x00208049, 0x00208200, 0x00208201, 0x00208208,
    0x00208209, 0x00208240, 0x00208241, 0x00208248, 0x00208249, 0x00209000, 0x00209001, 0x00209008, 0x00209009,
    0x00209040, 0x00209041, 0x00209048, 0x00209049, 0x00209200, 0x00209201, 0x00209208, 0x00209209, 0x00209240,
    0x00209241, 0x00209248, 0x00209249, 0x00240000, 0x00240001, 0x00240008, 0x00240009, 0x00240040, 0x00240041,
    0x00240048, 0x00240049, 0x00240200, 0x00240201, 0x00240208, 0x00240209, 0x00240240, 0x00240241, 0x00240248,
    0x00240249, 0x00241000, 0x00241001, 0x00241008, 0x00241009, 0x00241040, 0x00241041, 0x00241048, 0x00241049,
    0x00241200, 0x00241201, 0x00241208, 0x00241209, 0x00241240, 0x00241241, 0x00241248, 0x00241249, 0x00248000,
    0x00248001, 0x00248008, 0x00248009, 0x00248040, 0x00248041, 0x00248048, 0x00248049, 0x00248200, 0x00248201,
    0x00248208, 0x00248209, 0x00248240, 0x00248241, 0x00248248, 0x00248249, 0x00249000, 0x00249001, 0x00249008,
    0x00249009, 0x00249040, 0x00249041, 0x00249048, 0x00249049, 0x00249200, 0x00249201, 0x00249208, 0x00249209,
    0x00249240, 0x00249241, 0x00249248, 0x00249249};

const uint32_t kMortonCode256Y[256] = {
    0x00000000, 0x00000002, 0x00000010, 0x00000012, 0x00000080, 0x00000082, 0x00000090, 0x00000092, 0x00000400,
    0x00000402, 0x00000410, 0x00000412, 0x00000480, 0x00000482, 0x00000490, 0x00000492, 0x00002000, 0x00002002,
    0x00002010, 0x00002012, 0x00002080, 0x00002082, 0x00002090, 0x00002092, 0x00002400, 0x00002402, 0x00002410,
    0x00002412, 0x00002480, 0x00002482, 0x00002490, 0x00002492, 0x00010000, 0x00010002, 0x00010010, 0x00010012,
    0x00010080, 0x00010082, 0x00010090, 0x00010092, 0x00010400, 0x00010402, 0x00010410, 0x00010412, 0x00010480,
    0x00010482, 0x00010490, 0x00010492, 0x00012000, 0x00012002, 0x00012010, 0x00012012, 0x00012080, 0x00012082,
    0x00012090, 0x00012092, 0x00012400, 0x00012402, 0x00012410, 0x00012412, 0x00012480, 0x00012482, 0x00012490,
    0x00012492, 0x00080000, 0x00080002, 0x00080010, 0x00080012, 0x00080080, 0x00080082, 0x00080090, 0x00080092,
    0x00080400, 0x00080402, 0x00080410, 0x00080412, 0x00080480, 0x00080482, 0x00080490, 0x00080492, 0x00082000,
    0x00082002, 0x00082010, 0x00082012, 0x00082080, 0x00082082, 0x00082090, 0x00082092, 0x00082400, 0x00082402,
    0x00082410, 0x00082412, 0x00082480, 0x00082482, 0x00082490, 0x00082492, 0x00090000, 0x00090002, 0x00090010,
    0x00090012, 0x00090080, 0x00090082, 0x00090090, 0x00090092, 0x00090400, 0x00090402, 0x00090410, 0x00090412,
    0x00090480, 0x00090482, 0x00090490, 0x00090492, 0x00092000, 0x00092002, 0x00092010, 0x00092012, 0x00092080,
    0x00092082, 0x00092090, 0x00092092, 0x00092400, 0x00092402, 0x00092410, 0x00092412, 0x00092480, 0x00092482,
    0x00092490, 0x00092492, 0x00400000, 0x00400002, 0x00400010, 0x00400012, 0x00400080, 0x00400082, 0x00400090,
    0x00400092, 0x00400400, 0x00400402, 0x00400410, 0x00400412, 0x00400480, 0x00400482, 0x00400490, 0x00400492,
    0x00402000, 0x00402002, 0x00402010, 0x00402012, 0x00402080, 0x00402082, 0x00402090, 0x00402092, 0x00402400,
    0x00402402, 0x00402410, 0x00402412, 0x00402480, 0x00402482, 0x00402490, 0x00402492, 0x00410000, 0x00410002,
    0x00410010, 0x00410012, 0x00410080, 0x00410082, 0x00410090, 0x00410092, 0x00410400, 0x00410402, 0x00410410,
    0x00410412, 0x00410480, 0x00410482, 0x00410490, 0x00410492, 0x00412000, 0x00412002, 0x00412010, 0x00412012,
    0x00412080, 0x00412082, 0x00412090, 0x00412092, 0x00412400, 0x00412402, 0x00412410, 0x00412412, 0x00412480,
    0x00412482, 0x00412490, 0x00412492, 0x00480000, 0x00480002, 0x00480010, 0x00480012, 0x00480080, 0x00480082,
    0x00480090, 0x00480092, 0x00480400, 0x00480402, 0x00480410, 0x00480412, 0x00480480, 0x00480482, 0x00480490,
    0x00480492, 0x00482000, 0x00482002, 0x00482010, 0x00482012, 0x00482080, 0x00482082, 0x00482090, 0x00482092,
    0x00482400, 0x00482402, 0x00482410, 0x00482412, 0x00482480, 0x00482482, 0x00482490, 0x00482492, 0x00490000,
    0x00490002, 0x00490010, 0x00490012, 0x00490080, 0x00490082, 0x00490090, 0x00490092, 0x00490400, 0x00490402,
    0x00490410, 0x00490412, 0x00490480, 0x00490482, 0x00490490, 0x00490492, 0x00492000, 0x00492002, 0x00492010,
    0x00492012, 0x00492080, 0x00492082, 0x00492090, 0x00492092, 0x00492400, 0x00492402, 0x00492410, 0x00492412,
    0x00492480, 0x00492482, 0x00492490, 0x00492492};

const uint32_t kMortonCode256X[256] = {
    0x00000000, 0x00000004, 0x00000020, 0x00000024, 0x00000100, 0x00000104, 0x00000120, 0x00000124, 0x00000800,
    0x00000804, 0x00000820, 0x00000824, 0x00000900, 0x00000904, 0x00000920, 0x00000924, 0x00004000, 0x00004004,
    0x00004020, 0x00004024, 0x00004100, 0x00004104, 0x00004120, 0x00004124, 0x00004800, 0x00004804, 0x00004820,
    0x00004824, 0x00004900, 0x00004904, 0x00004920, 0x00004924, 0x00020000, 0x00020004, 0x00020020, 0x00020024,
    0x00020100, 0x00020104, 0x00020120, 0x00020124, 0x00020800, 0x00020804, 0x00020820, 0x00020824, 0x00020900,
    0x00020904, 0x00020920, 0x00020924, 0x00024000, 0x00024004, 0x00024020, 0x00024024, 0x00024100, 0x00024104,
    0x00024120, 0x00024124, 0x00024800, 0x00024804, 0x00024820, 0x00024824, 0x00024900, 0x00024904, 0x00024920,
    0x00024924, 0x00100000, 0x00100004, 0x00100020, 0x00100024, 0x00100100, 0x00100104, 0x00100120, 0x00100124,
    0x00100800, 0x00100804, 0x00100820, 0x00100824, 0x00100900, 0x00100904, 0x00100920, 0x00100924, 0x00104000,
    0x00104004, 0x00104020, 0x00104024, 0x00104100, 0x00104104, 0x00104120, 0x00104124, 0x00104800, 0x00104804,
    0x00104820, 0x00104824, 0x00104900, 0x00104904, 0x00104920, 0x00104924, 0x00120000, 0x00120004, 0x00120020,
    0x00120024, 0x00120100, 0x00120104, 0x00120120, 0x00120124, 0x00120800, 0x00120804, 0x00120820, 0x00120824,
    0x00120900, 0x00120904, 0x00120920, 0x00120924, 0x00124000, 0x00124004, 0x00124020, 0x00124024, 0x00124100,
    0x00124104, 0x00124120, 0x00124124, 0x00124800, 0x00124804, 0x00124820, 0x00124824, 0x00124900, 0x00124904,
    0x00124920, 0x00124924, 0x00800000, 0x00800004, 0x00800020, 0x00800024, 0x00800100, 0x00800104, 0x00800120,
    0x00800124, 0x00800800, 0x00800804, 0x00800820, 0x00800824, 0x00800900, 0x00800904, 0x00800920, 0x00800924,
    0x00804000, 0x00804004, 0x00804020, 0x00804024, 0x00804100, 0x00804104, 0x00804120, 0x00804124, 0x00804800,
    0x00804804, 0x00804820, 0x00804824, 0x00804900, 0x00804904, 0x00804920, 0x00804924, 0x00820000, 0x00820004,
    0x00820020, 0x00820024, 0x00820100, 0x00820104, 0x00820120, 0x00820124, 0x00820800, 0x00820804, 0x00820820,
    0x00820824, 0x00820900, 0x00820904, 0x00820920, 0x00820924, 0x00824000, 0x00824004, 0x00824020, 0x00824024,
    0x00824100, 0x00824104, 0x00824120, 0x00824124, 0x00824800, 0x00824804, 0x00824820, 0x00824824, 0x00824900,
    0x00824904, 0x00824920, 0x00824924, 0x00900000, 0x00900004, 0x00900020, 0x00900024, 0x00900100, 0x00900104,
    0x00900120, 0x00900124, 0x00900800, 0x00900804, 0x00900820, 0x00900824, 0x00900900, 0x00900904, 0x00900920,
    0x00900924, 0x00904000, 0x00904004, 0x00904020, 0x00904024, 0x00904100, 0x00904104, 0x00904120, 0x00904124,
    0x00904800, 0x00904804, 0x00904820, 0x00904824, 0x00904900, 0x00904904, 0x00904920, 0x00904924, 0x00920000,
    0x00920004, 0x00920020, 0x00920024, 0x00920100, 0x00920104, 0x00920120, 0x00920124, 0x00920800, 0x00920804,
    0x00920820, 0x00920824, 0x00920900, 0x00920904, 0x00920920, 0x00920924, 0x00924000, 0x00924004, 0x00924020,
    0x00924024, 0x00924100, 0x00924104, 0x00924120, 0x00924124, 0x00924800, 0x00924804, 0x00924820, 0x00924824,
    0x00924900, 0x00924904, 0x00924920, 0x00924924};

inline uint64_t PCCEncoder::mortonAddr( const int32_t x, const int32_t y, const int32_t z ) {
  uint64_t answer =
      kMortonCode256X[( x >> 16 ) & 0xFF] | kMortonCode256Y[( y >> 16 ) & 0xFF] | kMortonCode256Z[( z >> 16 ) & 0xFF];
  answer = answer << 24 | kMortonCode256X[( x >> 8 ) & 0xFF] | kMortonCode256Y[( y >> 8 ) & 0xFF] |
           kMortonCode256Z[( z >> 8 ) & 0xFF];
  answer = answer << 24 | kMortonCode256X[x & 0xFF] | kMortonCode256Y[y & 0xFF] | kMortonCode256Z[z & 0xFF];
  return answer;
}
uint64_t PCCEncoder::mortonAddr( const PCCPoint3D& vec, int depth ) {
  int x = int( vec.x() ) >> depth;
  int y = int( vec.y() ) >> depth;
  int z = int( vec.z() ) >> depth;
  return mortonAddr( x, y, z );
}
