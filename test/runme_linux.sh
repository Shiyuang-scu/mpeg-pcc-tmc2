#!/bin/bash

MAINDIR=$( dirname $( cd "$( dirname $0 )" && pwd ) );
# MAINDIR=$(dirname $(dirname $0));
EXTERNAL=$MAINDIR/dependencies/;

## Input parameters
SRCDIR=${MAINDIR}/mpeg_datasets/ # note: this directory must containt: http://mpegfs.int-evry.fr/MPEG/PCC/DataSets/pointCloud/CfP/datasets/Dynamic_Objects/People                            
CFGDIR=${MAINDIR}/cfg/
SEQ=25;       # in [22;26]
COND="C2AI";       # in [C2AI, C2LD, CWAI, CWRA]
RATE=1;       # in [1;5]
FRAMECOUNT=1;
THREAD=1;

## Set external tool paths
ENCODER=${MAINDIR}/bin/PccAppEncoder;
DECODER=${MAINDIR}/bin/PccAppDecoder;
HDRCONVERT=${EXTERNAL}HDRTools/build/bin/HDRConvert;
# VTMENCODER=${EXTERNAL}VTM/bin/EncoderApp;
# VTMDECODER=${EXTERNAL}VTM/bin/DecoderApp;
HMENCODER=${EXTERNAL}HM/bin/TAppEncoderStatic;
HMDECODER=${EXTERNAL}HM/bin/TAppDecoderStatic;

# ## set config path
# OCCMAPCONFIG=vtm/vtm-occupancy-map-ai.cfg #occupancyMapConfig
# GEOMAPCONFIG=vtm/vtm-geometry-ai-mp-separate-video.cfg #geometryMPConfig
# GEOCONGID=vtm/vtm-geometry-ai.cfg #geometryConfig
# ATTCONFIG=vtm/vtm-attribute-ai.cfg #attributeConfig

## Set Configuration based on sequence, condition and rate
if [ $COND == "C2AI" -o $COND == "C2RA" ] 
then
  case $SEQ in
      22) CFGSEQUENCE="sequence/queen.cfg";;
      23) CFGSEQUENCE="sequence/loot_vox10.cfg";;
      24) CFGSEQUENCE="sequence/redandblack_vox10.cfg";;
      25) CFGSEQUENCE="sequence/soldier_vox10.cfg";;
      26) CFGSEQUENCE="sequence/longdress_vox10.cfg";;
      27) CFGSEQUENCE="sequence/basketball_player_vox11";;
      28) CFGSEQUENCE="sequence/dancer_vox11.cfg";;
      *) echo "sequence not correct ($SEQ)";   exit -1;;
  esac
  case $RATE in
      5) CFGRATE="rate/ctc-r5.cfg";; 
      4) CFGRATE="rate/ctc-r4.cfg";; 
      3) CFGRATE="rate/ctc-r3.cfg";; 
      2) CFGRATE="rate/ctc-r2.cfg";; 
      1) CFGRATE="rate/ctc-r1.cfg";; 
      *) echo "rate not correct ($RATE)";   exit -1;;
  esac
  #BIN=mpeg_datasets/reconstruct/S${SEQ}${COND}R0${RATE}_F${FRAMECOUNT}.bin
  BIN=mpeg_datasets/down_reconstruct/S${SEQ}${COND}R0${RATE}_F${FRAMECOUNT}.bin
else
   case $SEQ in
      22) CFGSEQUENCE="sequence/queen-lossless.cfg";;
      23) CFGSEQUENCE="sequence/loot_vox10-lossless.cfg";;
      24) CFGSEQUENCE="sequence/redandblack_vox10-lossless.cfg";;
      25) CFGSEQUENCE="sequence/soldier_vox10-lossless.cfg";;
      26) CFGSEQUENCE="sequence/longdress_vox10-lossless.cfg";;
      27) CFGSEQUENCE="sequence/basketball_player_vox11-lossless";;
      28) CFGSEQUENCE="sequence/dancer_vox11-lossless.cfg";;
      *) echo "sequence not correct ($SEQ)";   exit -1;;
  esac 
  CFGRATE="rate/ctc-r5.cfg"
  BIN=mpeg_datasets/reconstruct/S${SEQ}${COND}_F${FRAMECOUNT}.bin
fi

case $COND in
  CWAI) CFGCOMMON="common/ctc-common-lossless-geometry-attribute.cfg";;
  CWLD) CFGCOMMON="common/ctc-common-lossless-geometry-attribute.cfg";; 
  C2AI) CFGCOMMON="common/ctc-common.cfg";;                           
  C2RA) CFGCOMMON="common/ctc-common.cfg";;           
  *) echo "Condition not correct ($COND)";   exit -1;;
esac

case $COND in
  CWAI) CFGCONDITION="condition/ctc-all-intra-lossless-geometry-attribute.cfg";;
  CWLD) CFGCONDITION="condition/ctc-low-delay-lossless-geometry-attribute.cfg";;
  C2AI) CFGCONDITION="condition/ctc-all-intra.cfg";;
  C2RA) CFGCONDITION="condition/ctc-random-access.cfg";;  
  *) echo "Condition not correct ($COND)";   exit -1;;
esac

# CFGCONDITION="condition/vtm-all-intra.cfg"
echo $CFGSEQUENCE

## Encoder 
#if [ ! -f $BIN ] 
#then 
  $ENCODER \
    --config=${CFGDIR}${CFGCOMMON} \
    --config=${CFGDIR}${CFGSEQUENCE} \
    --config=${CFGDIR}${CFGCONDITION} \
    --config=${CFGDIR}${CFGRATE} \
    --configurationFolder=${CFGDIR} \
    --uncompressedDataFolder=${SRCDIR} \
    --colorSpaceConversionPath=$HDRCONVERT \
    --nbThread=$THREAD \
    --keepIntermediateFiles=1 \
    --reconstructedDataPath=${BIN%.???}_rec_%04d.ply \
    --compressedStreamPath=${BIN} \
    --videoEncoderOccupancyPath=$HMENCODER \
    --videoEncoderGeometryPath=$HMENCODER \
    --videoEncoderAttributePath=$HMENCODER
#fi

## Decoder
$DECODER \
  --compressedStreamPath=$BIN \
  --colorSpaceConversionPath=${HDRCONVERT} \
  --inverseColorSpaceConversionConfig=${CFGDIR}hdrconvert/yuv420torgb444.cfg \
  --nbThread=$THREAD \
  --reconstructedDataPath=${BIN%.???}_dec_%04d.ply \
  --videoDecoderOccupancyPath=$HMDECODER \
  --videoDecoderGeometryPath=$HMDECODER \
  --videoDecoderAttributePath=$HMDECODER
