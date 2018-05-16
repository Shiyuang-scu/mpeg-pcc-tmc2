
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

namespace pcc {

class PCCPointSet3; 
class PCCGroupOfFrames;
class PCCBitstream;
class PCCContext;
class PCCFrameContext;
template <typename T, size_t N>
class PCCVideo;
typedef pcc::PCCVideo<uint8_t, 3> PCCVideo3B;
template <typename T, size_t N>
class PCCImage;
typedef pcc::PCCImage<uint8_t, 3> PCCImage3B;
struct PCCPatchSegmenter3Parameters; 

class PCCEncoder : public PCCCodec {
public:
  PCCEncoder();
  ~PCCEncoder();
  void setParameters( PCCEncoderParameters params );

  int compress( const PCCGroupOfFrames& sources, PCCContext &context, 
                PCCBitstream &bitstream, PCCGroupOfFrames& reconstructs );

private:
  int  compressHeader( PCCContext &context, PCCBitstream &bitstream );

  void compressOccupancyMap( PCCContext &context, PCCBitstream& bitstream );
  void compressOccupancyMap( PCCFrameContext &frame, PCCBitstream& bitstream );



  bool generateGeometryVideo( const PCCGroupOfFrames& sources, PCCContext &context );
  bool resizeGeometryVideo( PCCContext &context );
  bool dilateGeometryVideo( PCCContext &context );

  bool generateTextureVideo( const PCCGroupOfFrames& sources, PCCGroupOfFrames& reconstruct, PCCContext& context );




  void dilate( PCCFrameContext &frame, PCCImage3B &image, const PCCImage3B *reference = nullptr );
  void pack( PCCFrameContext& frame );
  void generateOccupancyMap( PCCFrameContext& frameContext );
  void printMap(std::vector<bool> img, const size_t sizeU, const size_t sizeV);
  void generateIntraImage( PCCFrameContext& frameContext, const size_t depthIndex, PCCImage3B &image);
  bool predictGeometryFrame( PCCFrameContext& frameContext, const PCCImage3B &reference, PCCImage3B &image);

  bool generateGeometryVideo( const PCCPointSet3& source, PCCFrameContext& frameContext,
                              const PCCPatchSegmenter3Parameters segmenterParams,
                              PCCVideo3B &videoGeometry );
  bool generateTextureVideo( const PCCPointSet3& reconstruct, PCCFrameContext& frameContext,
                             PCCVideo3B &video, const size_t frameCount );

  PCCEncoderParameters params_;

};

}; //~namespace

#endif /* PCCEncoder_h */
