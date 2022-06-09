from base64 import encode
import time
import subprocess
import datetime
from pathlib import Path
import logging
import yaml
from functools import partial
from multiprocessing import Pool, Manager
from tqdm import tqdm

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



    def encode_and_decode(self, enc_cmd, dec_cmd):

        encode_time = self.run_command(enc_cmd)
        decode_time = self.run_command(dec_cmd)

        return encode_time, decode_time

    def run_command(self, cmd):
        try: 

            # 1. execute command and record time
            start_time = time.time()
            #output = subprocess.run(cmd, capture_output=True, text=True)
            _ = subprocess.run(cmd)
            end_time = time.time()
            
             # 2. record the output information
            #timestamp = datetime.datetime.now().strftime('%Y%m%d_%H:%M:%S')
            #log_file = (Path(__file__).parents[0].joinpath(f'logs/execute_cmd_{timestamp}.log'))
            #with open(log_file, 'w') as f:
            #    f.write(output.stdout)

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
            '--computeMetrics=0',
            '--computeChecksum=0'
        ]
        return cmd

    def evaluate(self):
        pass


    def _run_process(self, src_dir, nor_dir, exp_dir):
        
        bin_file, out_file, evl_dir = (
            self._set_filepath(exp_dir)
        )

        enc_cmd = self.make_encode_cmd(src_dir, bin_file)
        dec_cmd = self.make_decode_cmd(bin_file, out_file)

        # 1. run encode and decode command
        encode_time, decode_time = self.encode_and_decode(enc_cmd, dec_cmd)

        # # 2. evaluate the results
        # self.evaluate(nor_pcfile, out_pc_file, bin_file, evl_dir, encode_time, decode_time)

    def run_experiment(self):
        
        src_dir = self._ds_cfg[self.ds_name]['dataset_dir']
        pattern = self._ds_cfg[self.ds_name]['test_pattern']
        # self.pc_files = glob_file(src_dir, pattern, verbose=True, fullpath=False)

        exp_dir = (
            Path('exps')
            .joinpath(f'{type(self).__name__}/{self.ds_name}/{self.rate}')
            .resolve()
        )


        logger.info(
            f"Start to run experiments on {self.ds_name} dataset "
            f"with {type(self).__name__} in {exp_dir}"
        )

        self._run_process(src_dir=src_dir, 
                        nor_dir=src_dir, 
                        exp_dir=exp_dir)

        # prun = partial(
        #     self._run_process,
        #     src_dir = src_dir, 
        #     nor_dir = src_dir, 
        #     exp_dir = exp_dir
        # )
        
        # parallel(prun, self.pc_files, self.nbprocesses)


    def _set_filepath(self, exp_dir):
        """Set up the experiment file paths, including encoded binary, 
        decoded point cloud, and evaluation log.
        
        Parameters
        ----------
        pcfile : `Union[str, Path]`
            The relative path of input point cloud.
        src_dir : `Union[str, Path]`
            The directory of input point cloud.
        nor_dir : `Union[str, Path]`
            The directory of input point cloud with normal. (Necessary 
            for p2plane metrics.)
        exp_dir : `Union[str, Path]`
            The directory to store experiments results.
        
        Returns
        -------
        `Tuple[str, str, str, str, str]`
            The full path of input point cloud, input point cloud with 
            normal, encoded binary file, output point cloud, and 
            evaluation log file.
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

        return (str(bin_file), str(out_file), str(evl_log))



if __name__ == '__main__':

    dataset_name = '8i_longdress'
    vpcc = VPCC(dataset_name)

    # for rate in range(5):
    #     vpcc.rate = f'r{rate+1}'
    #     vpcc.run_experiment()
    vpcc.rate = 'r2'
    vpcc.run_experiment()
