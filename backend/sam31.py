"""Serial local RPC bridge to an isolated SAM 3.1 WSL process."""
from pathlib import Path
import json
import queue
import subprocess
import tempfile
import threading
import time
import uuid
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
MARKER='SAM31_RPC:'


def linux_path(path):
    path=Path(path).resolve()
    return '/mnt/'+path.drive[0].lower()+path.as_posix()[2:]


class Sam31Engine:
    def __init__(self):
        self.status='not_loaded';self.error=None;self.device='cuda (WSL)'
        self.precision=None;self.process=None;self.reader=None;self.responses=queue.Queue()
        self.log=None;self.lock=threading.Lock()
        self.temp_dir=ROOT/'.cache'/'sam31-rpc';self.temp_dir.mkdir(parents=True,exist_ok=True)

    def _read(self,process,responses,log):
        try:
            for line in process.stdout:
                if line.startswith(MARKER):responses.put(json.loads(line[len(MARKER):]))
                else:log.write(line);log.flush()
        except Exception as exc:responses.put({'error':str(exc)})

    def _start(self):
        checkpoint=ROOT/'models'/'sam3.1'/'sam3.1_multiplex.pt'
        if not checkpoint.is_file():raise ValueError('D드라이브의 SAM 3.1 체크포인트가 없습니다.')
        runtime_file=ROOT/'.cache'/'sam31-runtime.json'
        if not runtime_file.is_file() and not (ROOT/'.venv-sam31-wsl'/'pyvenv.cfg').is_file():
            raise ValueError('SAM 3.1 전용 WSL 환경이 없습니다. setup-sam31.ps1을 실행하세요.')
        (ROOT/'logs').mkdir(exist_ok=True)
        self.log=(ROOT/'logs'/'sam31-worker.log').open('a',encoding='utf-8')
        base=linux_path(ROOT)
        runtime=json.loads(runtime_file.read_text(encoding='utf-8-sig')) if runtime_file.is_file() else {}
        command=['wsl.exe','-d',runtime.get('distribution','Ubuntu-24.04'),'--exec','env',
                 'PYTHONUNBUFFERED=1','PYTHONUTF8=1','HF_HUB_OFFLINE=1',
                 f'HF_HOME={base}/.cache/huggingface',f'TMPDIR={base}/.cache/tmp-linux',
                 f'TRITON_CACHE_DIR={base}/.cache/triton',f'CUDA_CACHE_PATH={base}/.cache/cuda-linux',
                 runtime.get('python',f'{base}/.venv-sam31-wsl/bin/python'),f'{base}/backend/sam31_worker.py']
        self.responses=queue.Queue()
        self.process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,
                                      text=True,encoding='utf-8',bufsize=1,
                                      creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        self.reader=threading.Thread(target=self._read,args=(self.process,self.responses,self.log),daemon=True)
        self.reader.start()

    def _request(self,operation,**payload):
        with self.lock:
            if self.process is None or self.process.poll() is not None:self._start()
            request_id=uuid.uuid4().hex
            self.process.stdin.write(json.dumps(dict(id=request_id,operation=operation,**payload))+'\n')
            self.process.stdin.flush();deadline=time.monotonic()+300
            while time.monotonic()<deadline:
                try:response=self.responses.get(timeout=1)
                except queue.Empty:
                    if self.process.poll() is not None:
                        raise RuntimeError('SAM 3.1 프로세스가 종료됐습니다. logs/sam31-worker.log를 확인하세요.')
                    continue
                if response.get('id')!=request_id:continue
                if response.get('error'):
                    if response['error'].startswith('OutOfMemoryError'):
                        raise RuntimeError('SAM 3.1 GPU 메모리가 부족합니다. 다른 GPU 작업을 종료하거나 SAM 2.1 모드를 선택하세요. 상세 로그: logs/sam31-worker.log')
                    raise RuntimeError(response['error'])
                return response['result']
            self.close()
            raise RuntimeError('SAM 3.1 처리 시간이 5분을 초과했습니다.')

    def load(self):
        if self.status=='ready' and self.process and self.process.poll() is None:return
        self.status='loading'
        try:
            result=self._request('load');self.precision=result['precision'];self.status='ready';self.error=None
        except Exception as exc:
            self.status='error';self.error=str(exc);self.close(keep_error=True);raise

    def _run(self,frames,masks,target=None,points=None,text=None):
        self.load()
        with tempfile.TemporaryDirectory(dir=self.temp_dir) as folder:
            folder=Path(folder);images=folder/'frames';images.mkdir()
            for i,frame in enumerate(frames):
                cv2.imwrite(str(images/f'{i:05d}.jpg'),frame.image,[cv2.IMWRITE_JPEG_QUALITY,95])
            np.savez_compressed(folder/'seeds.npz',**{str(k):v for k,v in masks.items()})
            metadata=self._request('infer',folder=linux_path(folder),target=target,points=points,text=text)
            with np.load(folder/'output.npz',allow_pickle=False) as arrays:
                output=[(frame,{int(key.split('_')[1]):arrays[key].copy() for key in arrays.files
                                if key.startswith(f'{i}_')}) for i,frame in enumerate(frames)]
            return output,metadata

    def infer(self,frames,masks,target=None,points=None):
        return self._run(frames,masks,target,points)[0]

    def segment_text(self,frame,text):
        outputs,metadata=self._run([frame],{},text=text)
        return outputs[0][1]

    def close(self,keep_error=False):
        process=self.process
        if process:
            if process.poll() is None:
                try:process.stdin.close();process.wait(timeout=8)
                except (OSError,subprocess.TimeoutExpired):process.terminate()
            if self.reader:self.reader.join(timeout=3)
            self.process=None
        if self.log:self.log.close();self.log=None
        if not keep_error:self.status='not_loaded';self.precision=None
