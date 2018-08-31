
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
#ifndef PCCEncoder_h
#define PCCEncoder_h

#include "PCCCommon.h"
#include "PCCEncoderParameters.h"
#include "PCCCodec.h"
#include "ArithmeticCodec.h"
#include "PCCMetadata.h"

namespace pcc {

class PCCPointSet3; 
class PCCGroupOfFrames;
class PCCBitstream;
class PCCContext;
class PCCFrameContext;
template <typename T, size_t N>
class PCCVideo;
typedef pcc::PCCVideo<uint8_t,  3> PCCVideoTexture;
typedef pcc::PCCVideo<uint16_t, 3> PCCVideoGeometry;
typedef pcc::PCCVideo<uint8_t,  3> PCCVideoOccupancyMap;
template <typename T, size_t N>
class PCCImage;
typedef pcc::PCCImage<uint8_t,  3> PCCImageTexture;
typedef pcc::PCCImage<uint16_t, 3> PCCImageGeometry;
typedef pcc::PCCImage<uint8_t,  3> PCCImageOccupancyMap;
struct PCCPatchSegmenter3Parameters;
class PCCPatch;
struct PCCBistreamPosition;
class Arithmetic_Codec;

class PCCEncoder : public PCCCodec {
public:
  PCCEncoder();
  ~PCCEncoder();
  void setParameters( PCCEncoderParameters params );

  int encode( const PCCGroupOfFrames& sources, PCCContext &context,
              PCCBitstream &bitstream, PCCGroupOfFrames& reconstructs );

private:
  int  writeMetadata(const PCCMetadata &metadata, PCCBitstream &bitstream );
  int  compressMetadata(const PCCMetadata &metadata, o3dgc::Arithmetic_Codec &arithmeticEncoder );

  int  compressHeader( PCCContext &context, PCCBitstream &bitstream );

  void compressOccupancyMap( PCCContext &context, PCCBitstream& bitstream );
  void compressOccupancyMap( PCCFrameContext &frame, PCCBitstream& bitstream, PCCFrameContext &preFrame, size_t frameIndex );
  void compressPatchMetaDataM42195(PCCFrameContext &frame, PCCFrameContext &preFrame, size_t numMatchedPatches, PCCBitstream &bitstream ,
    o3dgc::Arithmetic_Codec &arithmeticEncoder, o3dgc::Static_Bit_Model &bModel0, PCCBistreamPosition &startPosition);
  bool generateOccupancyMapVideo( const PCCGroupOfFrames& sources, PCCContext& context );
  bool generateOccupancyMapVideo(const size_t imageWidth, const size_t imageHeight,
                                 std::vector<uint32_t> &occupancyMap,
                                 PCCImageOccupancyMap &videoFrameOccupancyMap);

  bool generateGeometryVideo( const PCCGroupOfFrames& sources, PCCContext &context );
  bool resizeGeometryVideo( PCCContext &context );
  bool dilateGeometryVideo( PCCContext &context );

  bool generateTextureVideo( const PCCGroupOfFrames& sources, PCCGroupOfFrames& reconstruct, PCCContext& context );

  void writeMissedPointsGeometryNumber(PCCContext& context, PCCBitstream &bitstream);
  void writeMissedPointsTextureNumber(PCCContext& context, PCCBitstream &bitstream);
  
  void generateMissedPointsGeometryVideo(PCCContext& context, PCCGroupOfFrames& reconstructs);
  void generateMissedPointsTextureVideo(PCCContext& context, PCCGroupOfFrames& reconstructs, PCCBitstream &bitstream);
  
  void generateMPsGeometryImage    (PCCContext& context, PCCFrameContext& frame, PCCImageGeometry &image);
  void generateMPsTextureImage(PCCContext& context, PCCFrameContext& frame, PCCImageTexture &image, size_t shift, const PCCPointSet3& reconstruct);
  
  template <typename T>
  void dilate( PCCFrameContext &frame, PCCImage<T, 3> &image, const PCCImage<T, 3> *reference = nullptr );
  void pack( PCCFrameContext& frame );
  void packMissedPointsPatch( PCCFrameContext& frame, const std::vector<bool> &occupancyMap, size_t &width, 
                              size_t &height, size_t occupancySizeU, size_t occupancySizeV, size_t maxOccupancyRow);
  void spatialConsistencyPack(PCCFrameContext& frame, PCCFrameContext &prevFrame);
  void generateOccupancyMap( PCCFrameContext& frameContext );
  void printMap(std::vector<bool> img, const size_t sizeU, const size_t sizeV);
  void generateIntraImage( PCCFrameContext& frameContext, const size_t depthIndex, PCCImageGeometry &image);
  bool predictGeometryFrame( PCCFrameContext& frameContext, const PCCImageGeometry &reference, PCCImageGeometry &image);
  void generateMissedPointsPatch(const PCCPointSet3& source, PCCFrameContext& frameContext, bool useEnhancedDeltaDepthCode); //useEnhancedDeltaDepthCode for EDD
  void sortMissedPointsPatch(PCCFrameContext& frameContext);
  bool generateGeometryVideo( const PCCPointSet3& source, PCCFrameContext& frameContext,
                             const PCCPatchSegmenter3Parameters segmenterParams,
                             PCCVideoGeometry &videoGeometry, PCCFrameContext &prevFrame, size_t frameIndex);
  bool generateTextureVideo( const PCCPointSet3& reconstruct, PCCFrameContext& frameContext,
                             PCCVideoTexture &video, const size_t frameCount );
  //EDD
  void generateIntraEnhancedDeltaDepthImage(PCCFrameContext& frame, const PCCImageGeometry &imageRef, PCCImageGeometry &image);

  PCCEncoderParameters params_;
};

}; //~namespace

#endif /* PCCEncoder_h */
