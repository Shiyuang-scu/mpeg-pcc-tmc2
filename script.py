# TODO
# 1. eval results match and analyze
# 2. downsampling function
# 3. modify script.py
#   3.1 add '--keepIntermediateFiles=1'
#   3.2 output eval results in the .out file
#   3.3 add hausdorff distance metric


from base64 import encode
import time
import subprocess
import datetime
from pathlib import Path
import logging
import yaml
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm
import re
import os
import open3d as o3d

logger = logging.getLogger(__name__)

def parallel(func, filelist, use_gpu = False, nbprocesses = None):
    """Parallel processing with multiprocessing.Pool(), works better 
    with functools.partial().
    
    If ``use_gpu`` is True, ``gpu_queue`` will be passed to ``func`` as 
    a keyword argument. The input ``func`` needs to handle the keyword
    parameter ``gpu_queue`` and select the GPU with gpu_queue.get(). 
    Don't forget to put the GPU id back to the gpu_queue at the end of
    ``func``.
    
    Parameters
    ----------
    func : `Callable`
        The target function for parallel processing.
    filelist : `Iterable`
        The file list to process with the input function.
    use_gpu : `bool`, optional
        True for running NN-based PCC algs., False otherwise. 
        Defaults to False.
    nbprocesses : `int`, optional
        Specify the number of cpu parallel processes. If None, it will 
        equal to the cpu count. Defaults to None.
    
    Raises
    ------
    `ValueError`
        No available GPU.
    """
    assert len(filelist) > 0

    if use_gpu is True:
        # # Get the number of available GPUs
        # deviceIDs = GPUtil.getAvailable(
        #     order = 'first',
        #     limit = 8,
        #     maxLoad = 0.5,
        #     maxMemory = 0.5,
        #     includeNan=False,
        #     excludeID=[],
        #     excludeUUID=[]
        # )
        # process = len(deviceIDs)
        
        # if process <= 0:
        #     logger.error(
        #         "No available GPU. Check with the threshold parameters "
        #         "of ``GPUtil.getAvailable()``"
        #     )
        #     raise ValueError
        
        # manager = Manager()
        # gpu_queue = manager.Queue()
        # # gpu_queue = Queue()

        # for id in deviceIDs:
        #     gpu_queue.put(id)
        # pfunc = partial(func, gpu_queue=gpu_queue)
        pass
    else:
        process = nbprocesses
        pfunc = func

    with Pool(process) as pool:
        list(tqdm(pool.imap_unordered(pfunc, filelist), total=len(filelist)))

def load_cfg(cfg_file):
    """Load the PCC algs config file in YAML format with custom tag 
    !join.

    Parameters
    ----------
    cfg_file : `Union[str, Path]`
        The YAML config file.

    Returns
    -------
    `dict`
        A dictionary object loaded from the YAML config file.
    """
    # [ref.] https://stackoverflow.com/a/23212524
    ## define custom tag handler
    def join(loader, node):
        seq = loader.construct_sequence(node)
        return ''.join([str(i) for i in seq])

    ## register the tag handler
    yaml.add_constructor('!join', join)
    
    with open(cfg_file, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    return cfg

def glob_file(src_dir, pattern, fullpath=False, verbose=False):
    """Recursively glob the files in ``src_dir`` with input ``pattern``.

    Parameters
    ----------
    src_dir : `Union[str, Path]`
        The root directory to glob the files.
    pattern : `str`
        The pattern to glob the files.
    fullpath : `bool`, optional
        True for full path of files, False for filename only, 
        by default False
    verbose : `bool`, optional
        True to log message, False otherwise, by default False

    Returns
    -------
    `List[Path]`
        Files that match the glob pattern.

    Raises
    ------
    `ValueError`
        No any file match pattern in `src_dir`.
    """
    if fullpath:
        files = list(p.resolve(True) for p in Path(src_dir).rglob(pattern))
    else:
        files = list(
            p.relative_to(src_dir) for p in Path(src_dir).rglob(pattern)
        )
    
    if len(files) <= 0:
        logger.error(
            f"Not found any files "
            f"with pattern: {pattern} in {Path(src_dir).resolve(True)}")
        raise ValueError
    
    if verbose:
        logger.info(
            f"Found {len(files)} files "
            f"with pattern: {pattern} in {Path(src_dir).resolve(True)}")

    return files

class VPCC:
    def __init__(self, ds_name):
        self.ds_name = ds_name

        algs_cfg_file = (
            Path(__file__).parents[0]
            .joinpath(f'cfg/{type(self).__name__}.yml').resolve()
        )
        ds_cfg_file = (
            Path(__file__).parents[0]
            .joinpath('cfg/datasets.yml').resolve()
        )
        
        self._ds_cfg = load_cfg(ds_cfg_file)
        self._algs_cfg = load_cfg(algs_cfg_file)

        self.nbprocesses = None

    def _set_filepath(self, exp_dir, nor_dir):
        """Set up the experiment file paths, including encoded binary, 
        decoded point cloud, and evaluation log.
        
        Parameters
        ----------
        exp_dir : `Union[str, Path]`
            The directory to store experiments results.
        
        """
        bin_file = (
            Path(exp_dir)
            .joinpath('bin', self.ds_name).with_suffix(self._algs_cfg['bin_suffix'])
        )
        out_file = Path(exp_dir).joinpath('dec', '%04d.ply')
        evl_log = Path(exp_dir).joinpath('evl', self.ds_name).with_suffix('.log')

        bin_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        evl_log.parent.mkdir(parents=True, exist_ok=True)


        nor_file = Path(nor_dir).joinpath(f'{self.ds_name}_vox10_%04d.ply')


        return (str(bin_file), str(out_file), str(evl_log), str(nor_file))

    def encode_and_decode(self, enc_cmd, dec_cmd):

        encode_time = self.run_command(enc_cmd)
        decode_time = self.run_command(dec_cmd)

        return encode_time, decode_time

    def run_command(self, cmd):
        try: 

            # 1. execute command and record time
            start_time = time.time()
            # output = subprocess.run(cmd, capture_output=True, text=True)
            _ = subprocess.run(cmd)
            end_time = time.time()
            
            # # 2. record the output information
            # timestamp = datetime.datetime.now().strftime('%Y%m%d_%H:%M:%S')
            # log_file = (Path(__file__).parents[0].joinpath(f'logs/execute_cmd_{timestamp}.log'))
            # with open(log_file, 'w') as f:
            #     f.write(output.stdout)

        except subprocess.CalledProcessError as e:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H:%M:%S')
            log_file = (Path(__file__).parents[0].joinpath(f'logs/execute_cmd_error_{timestamp}.log'))
            
            with open(log_file, 'w') as f:
                lines = [
                    f"The stdout and stderr of executed command: ",
                    f"{''.join(str(s)+' ' for s in cmd)}",
                    "\n",
                    "===== stdout =====",
                    f"{e.stdout}",
                    "\n",
                    "===== stderr =====",
                    f"{e.stderr}",
                ]
                f.writelines('\n'.join(lines))

            logger.error(
                f"Error occurs when executing command: "
                "\n"
                f"{''.join(str(s)+' ' for s in cmd)}"
                "\n"
                f"Check {log_file} for more informations."
            )
            

        return end_time - start_time

    def make_encode_cmd(self, src_dir, bin_file):
        cmd = [
            self._algs_cfg['encoder'],
            f'--uncompressedDataFolder={src_dir}',
            f'--compressedStreamPath={bin_file}',
            '--configurationFolder=cfg/',
            '--config=cfg/common/ctc-common.cfg',
            '--keepIntermediateFiles=1',
            '--nbThread=5'，
            f'--config={self._ds_cfg[self.ds_name]["dataset_cfg"]}',
            f'--config={self._algs_cfg["condition_cfg"]}',
            f'--config={self._algs_cfg[self.rate]["rate_cfg"]}',
            f'--videoEncoderOccupancyPath={self._algs_cfg["videoEncoder"]}',
            f'--videoEncoderGeometryPath={self._algs_cfg["videoEncoder"]}',
            f'--videoEncoderAttributePath={self._algs_cfg["videoEncoder"]}',
            '--computeMetrics=0',
            '--computeChecksum=0'
        ]
        
        return cmd

    def make_decode_cmd(self, bin_file, out_file):
        cmd = [
            self._algs_cfg['decoder'],
            f'--compressedStreamPath={bin_file}',
            f'--reconstructedDataPath={out_file}',
            f'--videoDecoderOccupancyPath={self._algs_cfg["videoDecoder"]}',
            f'--videoDecoderGeometryPath={self._algs_cfg["videoDecoder"]}',
            f'--videoDecoderAttributePath={self._algs_cfg["videoDecoder"]}',
            f'--inverseColorSpaceConversionConfig={self._algs_cfg["inverseColorSpaceConversionConfig"]}',
            f'--startFrameNumber={self._ds_cfg[self.ds_name]["startFrameNumber"]}'
            '--computeMetrics=0',
            '--computeChecksum=0'
        ]
        return cmd

    def _rand_down_samp_pcd(self, absolute_path, sample_rate):
        """
        perform random downsampling on point cloud data file (.ply).
        
        :param absolute_path: absolute path of point cloud data file
        :param sample_rate: scale ratio of random downsampling
        """

        pcd = o3d.io.read_point_cloud(absolute_path)
        down_pcd = pcd.random_down_sample(float(sample_rate))

        return down_pcd

    def _save_down_pcd(self, down_pcd, absolute_path):
        """
        save the downsampled point cloud data file (.ply).
        
        :param down_pcd: downsampled point cloud data file
        :param absolute_path: absolute path of point cloud data file
        """

        o3d.io.write_point_cloud(absolute_path, down_pcd, write_ascii=True)

    def _downsampling(self, src_root, scale_ratio):
        output_dir = Path(src_root).joinpath('down_Ply', scale_ratio)
        input_dir = str(Path(src_root).joinpath('Ply'))+'/'

        if output_dir.is_dir():
            pass
        else:
            output_dir.mkdir(parents=True, exist_ok=True)

            for idx, filename in enumerate(os.listdir(input_dir)):
                if idx % 50 == 0:
                    print(f"Already processed {idx} files")
                if filename.endswith(".ply"): 
                    # 1. random downsampling
                    input_path = os.path.join(input_dir, filename)
                    down_pcd = self._rand_down_samp_pcd(input_path, scale_ratio)
                    output_path = str(output_dir.joinpath(filename))
                    self._save_down_pcd(down_pcd, output_path)

                    # 2. change the data format
                    with open(output_path, 'r') as f:
                        filedata = f.read()
                    filedata = filedata.replace(' double', ' float')
                    with open(output_path, 'w') as f:
                        f.write(filedata)


    def _evaluate_and_log(self, ref_path, out_path, bin_file, evl_log, enc_time, dec_time):
        
        startFrameNumber = self._ds_cfg[self.ds_name]["startFrameNumber"]

        evaluator = Evaluator(ref_path, out_path, startFrameNumber, bin_file, enc_time, dec_time)
        eva_result = evaluator.evaluate()
        
        with open(evl_log, 'w') as f:
            f.write(eva_result)

    def _run_process(self, src_dir, nor_dir, exp_dir):


        bin_file, out_file, evl_log, nor_file = (
            self._set_filepath(exp_dir, nor_dir)
        )
        # 1. run encode and decode command
        enc_cmd = self.make_encode_cmd(src_dir, bin_file)
        dec_cmd = self.make_decode_cmd(bin_file, out_file)
        encode_time, decode_time = self.encode_and_decode(enc_cmd, dec_cmd)

        # # 2. evaluate the results
        self._evaluate_and_log(nor_file, out_file, bin_file, evl_log, encode_time, decode_time)

    def run_experiment(self):

        is_downsamp = self._algs_cfg['is_downsampling']
        src_root = self._ds_cfg[self.ds_name]['dataset_dir']
        
        # downsampling with specific scale ratio
        if is_downsamp:
            # set filepath
            scale_ratio = str(self._algs_cfg['scale_ratio'])
            src_dir = str(Path(src_root).joinpath('down_Ply', scale_ratio))+'/'
            exp_dir = (
                Path('exps')
                .joinpath(f'{type(self).__name__}/{self.ds_name}/{self.rate}/{scale_ratio}/')
                .resolve()
            )
            # perform downsampling
            self._downsampling(src_root, scale_ratio)

        else:
            src_dir = str(Path(src_root).joinpath('Ply'))+'/'
            exp_dir = (
                Path('exps')
                .joinpath(f'{type(self).__name__}/{self.ds_name}/{self.rate}/')
                .resolve()
            )


        logger.info(
            f"Start to run experiments on {self.ds_name} dataset "
            f"with {type(self).__name__} in {exp_dir}"
        )

        # prun = partial(
        #     self._run_process,
        #     src_dir = src_dir, 
        #     nor_dir = src_dir, 
        #     exp_dir = exp_dir
        # )
        
        # parallel(prun, self.pc_files, self.nbprocesses)

        self._run_process(src_dir=src_dir, 
                nor_dir=src_dir, 
                exp_dir=exp_dir
                )

class Evaluator:
    def __init__(self, src_path, out_path, startFrameNumber, bin_file, enc_time, dec_time,):
        self._ref_path = src_path
        self._target_path = out_path

        # self._ref_files = sorted([str(f) for f in self._ref_dir.iterdir() if f.is_file()])
        # self._target_files = sorted([str(f) for f in self._target_dir.iterdir() if f.is_file()])

        self._bin_file = Path(bin_file) if bin_file else None
        self._enc_t = enc_time
        self._dec_t = dec_time
        self._startFrameNumber = startFrameNumber
        self._results = ''

        # self.ref_pc_size = 0

    def evaluate(self):

        # 1. log metrics result
        # file size of reference point cloud in `kB`
        # self.ref_pc_size += (Path(self._ref_pc).stat().st_size / 1000)

        # ProjMetrics = ProjectionBasedMetrics(self._ref_pc, self._target_pc, self._o3d_vis)
        PointMetrics = PointBasedMetrics(self._ref_path, self._target_path, self._startFrameNumber)
        
        # self._results += ProjMetrics.evaluate()
        self._results += PointMetrics.evaluate()
        
        # 2. log running time, file size, and bitrate
        self._log_running_time_and_bitrate()
        
        return self._results

    def _log_running_time_and_bitrate(self) -> None:
        """Log running time (encoding and decoding) and encoded 
        binary size.
        """
        
        # check if binary file is initialized in constructor
        if self._bin_file:
            # file size of compressed binary file in `kB`
            bin_size = Path(self._bin_file).stat().st_size / 1000
            # compression_ratio = bin_size / ref_pc_size  # kB
            kbps = (bin_size) / 10 # 10 seconds
        else:
            bin_size = compression_ratio = kbps = -1

        # check if running time is initialized in constructor
        if self._enc_t and self._dec_t:
            enc_t = f"{self._enc_t:0.4f}"
            dec_t = f"{self._dec_t:0.4f}"
        else:
            enc_t = dec_t = -1

        lines = [
            f"========== Time & Binary Size ==========",
            f"Encoding time (s)           : {enc_t}",
            f"Decoding time (s)           : {dec_t}",
            # f"Source point cloud size (kB): {ref_pc_size}",
            f"Total binary files size (kB): {bin_size}",
            # f"Compression ratio           : {compression_ratio}",
            f"kbps (kb per second)        : {kbps}",
            "\n",
        ]
        lines = '\n'.join(lines)

        self._results += lines

class PointBasedMetrics:
    """Class for evaluating view independent metrics of given point 
    clouds.
    
    View Independent Metrics:
        ACD (1->2), ACD (2->1),
        CD,
        CD-PSNR,
        Hausdorff,
        Y-CPSNR, U-CPSNR, V-CPSNR,
        Hybrid geo-color
    """
    
    def __init__(self, ref_path, target_path, startFrameNumber):
        
        self._ref_path = ref_path
        self._target_path = target_path
        self._startFrameNumber = startFrameNumber

        self._results = []

        self.frameCount = len(list(Path(self._target_path).parents[0].iterdir()))

        self.METRIC = (
            Path(__file__).parents[0].
            joinpath("bin/PccAppMetrics")
            .resolve()
        )

    def evaluate(self) -> str:
        """Run the evaluation and generate the formatted evaluation 
        results.
        
        Parameters
        ----------
        ref_pc : `Union[str, Path]`
            Full path of the reference point cloud. Use point cloud with
            normal to calculate the p2plane metrics.
        target_pc : `Union[str, Path]`
            Full path of the target point.
        color : `bool`, optional
            True for calculating color metric, false otherwise. Defaults
            to false.
        resolution : `int`, optional
            Maximum NN distance of the ``ref_pc``. If the resolution is 
            not specified, it will be calculated on the fly. Defaults to
            None.
        enc_t : `float`, optional
            Total encoding time. Defaults to None.
        dec_t : `float`, optional
            Total decoding time. Defaults to None.
        bin_files : `List[Union[str, Path]]`, optional
            List of the full path of the encoded binary file. Used for 
            calculate the compression ratio and bpp.
        
        Returns
        -------
        `str`
            The formatted evaluation results.
        """
        
        self._get_quality_metrics()
        
        # ret = '\n'.join(self._results)
        ret = ''.join(self._results)
        
        return ret

    def _get_quality_metrics(self)-> None:
        """Calculate and parse the results of quality metrics from
        pc_error.
        """
        ret = self._metric_wrapper()
        logger.info(
            f"{ret}"
        )

        chosen_metrics = [
        r'mseF      \(p2point\): (\d+\.\d+)',
        r'mseF,PSNR \(p2point\): (\d+\.\d+)',
        r'mseF      \(p2plane\): (\d+\.\d+)',
        r'mseF,PSNR \(p2plane\): (\d+\.\d+)',
        r'h.       F\(p2point\): (\d+\.\d+)',
        r'h.,PSNR  F\(p2point\): (\d+\.\d+)',
        r'h.       F\(p2plane\): (\d+\.\d+)',
        r'h.,PSNR  F\(p2plane\): (\d+\.\d+)',
        r'c\[0\],    F         : (\d+\.\d+)',
        r'c\[1\],    F         : (\d+\.\d+)',
        r'c\[2\],    F         : (\d+\.\d+)',
        r'c\[0\],PSNRF         : (\d+\.\d+)',
        r'c\[1\],PSNRF         : (\d+\.\d+)',
        r'c\[2\],PSNRF         : (\d+\.\d+)'
        ]

        temp = []
        found_val = []

        for idx, pattern in enumerate(chosen_metrics):
            temp = []
            for line in ret.splitlines():
                m = re.search(pattern, line)
                if m:
                    temp.append(float(m.group(1)))
            found_val.append("{:.5f}".format(sum(temp) / (len(temp)+0.001)))
        
        lines = [
            "========== Point-based Metrics =========",
            f'mse      (p2point): {found_val[0]}',
            f'mse,PSNR (p2point): {found_val[1]}',
            f'mse      (p2plane): {found_val[2]}',
            f'mse,PSNR (p2plane): {found_val[3]}',
            f'h.       (p2point): {found_val[4]}',
            f'h.,PSNR  (p2point): {found_val[5]}',
            f'h.       (p2plane): {found_val[6]}',
            f'h.,PSNR  (p2plane): {found_val[7]}',
            f'c[0],             : {found_val[8]}',
            f'c[1],             : {found_val[9]}',
            f'c[2],             : {found_val[10]}',
            f'c[0],PSNR         : {found_val[11]}',
            f'c[1],PSNR         : {found_val[12]}',
            f'c[2],PSNR         : {found_val[13]}',
            "\n"
            ]
        
        self._results += '\n'.join(lines)
        # self._results += ret

    def _metric_wrapper(self) -> str:
        """Wrapper of the metric software, which modifies the formulas 
        and adds new metrics based on mpeg-pcc-dmetric.

        Returns
        -------
        `str`
            The result of objective quality metrics.
        """

        cmd = [
            self.METRIC,
            f'--uncompressedDataPath={self._ref_path}',
            f'--reconstructedDataPath={self._target_path}',
            f'--frameCount={self.frameCount}',
            f'--startFrameNumber={self._startFrameNumber}'
        ]

        ret = subprocess.run(cmd, capture_output=True, universal_newlines=True)

        return ret.stdout


if __name__ == '__main__':

    # temporarily applied to downsampling the files in linux server


    dataset_name = 'longdress'
    vpcc = VPCC(dataset_name)

    for rate in range(5):
        vpcc.rate = f'r{rate+1}'
        vpcc.run_experiment()
    # vpcc.rate = 'r2'
    # vpcc.run_experiment()