
#include "PCCSHMAppVideoEncoder.h"
#include "PCCSystem.h"

#ifdef USE_SHMAPP_VIDEO_CODEC

using namespace pcc;

template <typename T>
PCCSHMAppVideoEncoder<T>::PCCSHMAppVideoEncoder() {}
template <typename T>
PCCSHMAppVideoEncoder<T>::~PCCSHMAppVideoEncoder() {}

std::string replace( std::string string, const std::string& from, const std::string& to ) {
  size_t position = 0;
  while ( ( position = string.find( from, position ) ) != std::string::npos ) {
    string.replace( position, from.length(), to );
    position += to.length();
  }
  return string;
}

template <typename T>
void PCCSHMAppVideoEncoder<T>::encode( PCCVideo<T, 3>&            videoSrc,
                                       PCCVideoEncoderParameters& params,
                                       PCCVideoBitstream&         bitstream,
                                       PCCVideo<T, 3>&            videoRec ) {
  int32_t numLayers = std::stoi( getParameterFromConfigurationFile( params.encoderConfig_, "NumLayers" ) );

  const size_t width      = videoSrc.getWidth();
  const size_t height     = videoSrc.getHeight();
  const size_t frameCount = videoSrc.getFrameCount();
  if ( numLayers >= 1 ) {
    // Set Layer size and src and rec raw video names
    std::vector<std::string> srcYuvFileName, recYuvFileName;
    std::vector<size_t>      widthLayers, heightLayers;
    for ( size_t i = 0; i < numLayers; i++ ) {
      widthLayers.push_back( width / ( params.shvcRateX_ * ( numLayers - i ) ) );
      heightLayers.push_back( height / ( params.shvcRateY_ * ( numLayers - i ) ) );
      srcYuvFileName.push_back( replace( params.srcYuvFileName_, stringFormat( "_%dx%d_", width, height ),
                                         stringFormat( "_%dx%d_", widthLayers[i], heightLayers[i] ) ) );
      recYuvFileName.push_back( replace( params.recYuvFileName_, stringFormat( "_%dx%d_", width, height ),
                                         stringFormat( "_%dx%d_", widthLayers[i], heightLayers[i] ) ) );
    }

    // SHVC khu H video downsampling
    std::vector<PCCVideo<T, 3>> videoSrcLayers;
    // SHVC khu H using setValue and getValue in YUV420 error channelIndex 1&2
    if ( numLayers >= 1 && params.shvcRateX_ >= 2 && params.shvcRateY_ >= 2 ) {
      std::cout << "SHVC sub video generate" << std::endl;
      for ( size_t i = 0; i < numLayers; i++ ) {
        int32_t        scaleX = params.shvcRateX_ * ( numLayers - i );
        int32_t        scaleY = params.shvcRateY_ * ( numLayers - i );
        PCCVideo<T, 3> videoDst;
        videoDst.resize( frameCount );
        for ( size_t j = 0; j < videoDst.getFrameCount(); j++ ) {
          auto& imageSrc = videoSrc.getFrame( j );
          auto& imageDst = videoDst.getFrame( j );
          imageDst.resize( width / scaleX, height / scaleY, imageSrc.getColorFormat() );
          for ( size_t v = 0; v < height; v += scaleY ) {
            for ( size_t u = 0; u < width; u += scaleX ) {
              imageDst.setValue( 0, u / scaleX, v / scaleY, imageSrc.getValue( 0, u, v ) );
              imageDst.setValue( 1, u / scaleX, v / scaleY, imageSrc.getValue( 1, u, v ) );
              imageDst.setValue( 2, u / scaleX, v / scaleY, imageSrc.getValue( 2, u, v ) );
            }
          }
        }
        videoSrcLayers.push_back( videoDst );
        videoDst.write( srcYuvFileName[i], params.inputBitDepth_ == 8 ? 1 : 2  ); 
      }
    }

    std::stringstream cmd;
    cmd << params.encoderPath_;
    cmd << " -c " << params.encoderConfig_;
    cmd << " --InputChromaFormat=" << ( params.use444CodecIo_ ? "444" : "420" );
    cmd << " --FramesToBeEncoded=" << frameCount;
    cmd << " --FrameSkip=0";
    cmd << " --BitstreamFile=" << params.binFileName_;

    for ( size_t i = 0; i < numLayers; i++ ) {
      cmd << " --InputFile" << std::to_string( i ) << "=" << srcYuvFileName[i];
      cmd << " --InputBitDepth" << std::to_string( i ) << "=" << params.inputBitDepth_;
      cmd << " --OutputBitDepth" << std::to_string( i ) << "=" << params.outputBitDepth_;
      cmd << " --FrameRate" << std::to_string( i ) << "=30";
      cmd << " --SourceWidth" << std::to_string( i ) << "=" << widthLayers[i];
      cmd << " --SourceHeight" << std::to_string( i ) << "=" << heightLayers[i];
      cmd << " --ReconFile" << std::to_string( i ) << "=" << recYuvFileName[i];
      cmd << " --QP" << std::to_string( i ) << "=" << params.qp_;
    }
    cmd << " --InputFile" << std::to_string( numLayers ) << "=" << params.srcYuvFileName_;
    cmd << " --InputBitDepth" << std::to_string( numLayers ) << "=" << params.inputBitDepth_;
    cmd << " --OutputBitDepth" << std::to_string( numLayers ) << "=" << params.outputBitDepth_;
    cmd << " --FrameRate" << std::to_string( numLayers ) << "=30";
    cmd << " --SourceWidth" << std::to_string( numLayers ) << "=" << width;
    cmd << " --SourceHeight" << std::to_string( numLayers ) << "=" << height;
    cmd << " --ReconFile" << std::to_string( numLayers ) << "=" << params.recYuvFileName_;
    cmd << " --QP" << std::to_string( numLayers ) << "=" << params.qp_;
    cmd << " --FrameSkip=0";
    std::cout << cmd.str() << std::endl;

    videoSrc.write( params.srcYuvFileName_, params.inputBitDepth_ == 8 ? 1 : 2 );
    for ( size_t i = 0; i < numLayers; i++ ) {
      videoSrcLayers[i].write( recYuvFileName[i], params.inputBitDepth_ == 8 ? 1 : 2 );
    }
    if ( pcc::system( cmd.str().c_str() ) ) {
      std::cout << "Error: can't run system command!" << std::endl;
      exit( -1 );
    }

    PCCCOLORFORMAT format = getColorFormat( params.recYuvFileName_ );
    videoRec.clear();
    std::cout << "videoRec.clear();" << std::endl;
    videoRec.read( params.recYuvFileName_, width, height, format, params.outputBitDepth_ == 8 ? 1 : 2 );
    std::cout << "videoRec.read" << std::endl;
    // for ( size_t i = 0; i < numLayers; i++ ) {
    //   PCCVideo<T, 3> video_temp;
    //   video_temp.read( params.sub_recYuvFileName_[i], sub_width[i], sub_height[i], format, frameCount,
    //                    params.outputBitDepth_ == 8 ? 1 : 2 );
    //   sub_videoRec.push_back( video_temp );
    // }

    bitstream.read( params.binFileName_ );
  } else {
    std::stringstream cmd;
    cmd << params.encoderPath_;
    cmd << " -c " << params.encoderConfig_;
    cmd << " --InputFile=" << params.srcYuvFileName_;
    cmd << " --InputBitDepth=" << params.inputBitDepth_;
    cmd << " --InputChromaFormat=" << ( params.use444CodecIo_ ? "444" : "420" );
    cmd << " --OutputBitDepth=" << params.outputBitDepth_;
    cmd << " --OutputBitDepthC=" << params.outputBitDepth_;
    cmd << " --FrameRate=30";
    cmd << " --FrameSkip=0";
    cmd << " --SourceWidth=" << width;
    cmd << " --SourceHeight=" << height;
    cmd << " --ConformanceWindowMode=1 ";
    cmd << " --FramesToBeEncoded=" << frameCount;
    cmd << " --BitstreamFile=" << params.binFileName_;
    cmd << " --ReconFile=" << params.recYuvFileName_;
    cmd << " --QP=" << params.qp_;
    if ( params.transquantBypassEnable_ != 0 ) { cmd << " --TransquantBypassEnable=1"; }
    if ( params.cuTransquantBypassFlagForce_ != 0 ) { cmd << " --CUTransquantBypassFlagForce=1"; }
    if ( params.internalBitDepth_ != 0 ) {
      cmd << " --InternalBitDepth=" << params.internalBitDepth_;
      cmd << " --InternalBitDepthC=" << params.internalBitDepth_;
    }
    if ( params.usePccMotionEstimation_ ) {
      cmd << " --UsePccMotionEstimation=1";
      cmd << " --BlockToPatchFile=" << params.blockToPatchFile_;
      cmd << " --OccupancyMapFile=" << params.occupancyMapFile_;
      cmd << " --PatchInfoFile=" << params.patchInfoFile_;
    }
    if ( params.inputColourSpaceConvert_ ) { cmd << " --InputColourSpaceConvert=RGBtoGBR"; }

    std::cout << cmd.str() << std::endl;

    videoSrc.write( params.srcYuvFileName_, params.inputBitDepth_ == 8 ? 1 : 2 );
    if ( pcc::system( cmd.str().c_str() ) ) {
      std::cout << "Error: can't run system command!" << std::endl;
      exit( -1 );
    }
    PCCCOLORFORMAT format = getColorFormat( params.recYuvFileName_ );
    videoRec.clear();
    videoRec.read( params.recYuvFileName_, width, height, format, params.outputBitDepth_ == 8 ? 1 : 2 );
    bitstream.read( params.binFileName_ );
  }
}

template <typename T>
PCCCOLORFORMAT PCCSHMAppVideoEncoder<T>::getColorFormat( std::string& name ) {
  if ( ( name.find( "_p444.rgb" ) ) != std::string::npos ) {
    return PCCCOLORFORMAT::RGB444;
  } else if ( ( name.find( "_p444.yuv" ) ) != std::string::npos ) {
    return PCCCOLORFORMAT::YUV444;
  } else if ( ( name.find( "_p420.yuv" ) ) != std::string::npos ) {
    return PCCCOLORFORMAT::YUV420;
  } else {
    printf( "PCCSHMAppVideoEncoder can't find parameters %s \n", name.c_str() );
    exit( -1 );
  }
  return PCCCOLORFORMAT::UNKNOWN;
}

template class pcc::PCCSHMAppVideoEncoder<uint8_t>;
template class pcc::PCCSHMAppVideoEncoder<uint16_t>;

#endif  //~USE_SHMAPP_VIDEO_CODEC
