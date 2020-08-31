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

#ifdef USE_JMAPP_VIDEO_CODEC

#include "PCCJMAppVideoDecoder.h"
#include "PCCSystem.h"
#include "PccAvcParser.h"

using namespace pcc;

template <typename T>
PCCJMAppVideoDecoder<T>::PCCJMAppVideoDecoder() {}
template <typename T>
PCCJMAppVideoDecoder<T>::~PCCJMAppVideoDecoder() {}

template <typename T>
void PCCJMAppVideoDecoder<T>::decode( PCCVideoBitstream& bitstream,
                                      size_t             outputBitDepth,
                                      bool               RGB2GBR,
                                      PCCVideo<T, 3>&    video,
                                      const std::string& decoderPath,
                                      const std::string& fileName,
                                      const size_t       frameCount,
                                      const size_t       codecId ) {
  size_t       width = 0, height = 0;
  PccAvcParser avcParser;
  avcParser.getVideoSize( bitstream.vector(), width, height, codecId );

  const std::string binFileName = fileName + ".bin";
  const std::string reconFile =
      addVideoFormat( fileName + "_rec", width, height, !RGB2GBR, !RGB2GBR, outputBitDepth == 10 ? "10" : "8" );
  bitstream.write( binFileName );

  std::stringstream cmd;
  cmd << decoderPath << " -i " << binFileName << " -o " << reconFile;
  std::cout << cmd.str() << '\n';
  if ( pcc::system( cmd.str().c_str() ) ) {
    std::cout << "Error: can't run system command!" << std::endl;
    exit( -1 );
  }
  PCCCOLORFORMAT format = RGB2GBR ? PCCCOLORFORMAT::RGB444 : PCCCOLORFORMAT::YUV420;
  video.clear();
  video.read( reconFile, width, height, format, frameCount, outputBitDepth == 8 ? 1 : 2 );
  printf( "File read size = %zu x %zu frame count = %zu \n", video.getWidth(), video.getHeight(),
          video.getFrameCount() );

  removeFile( binFileName );
  removeFile( reconFile );
}

template class pcc::PCCJMAppVideoDecoder<uint8_t>;
template class pcc::PCCJMAppVideoDecoder<uint16_t>;

#endif