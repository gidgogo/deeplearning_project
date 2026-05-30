# Colab 실행 가이드 (2시간 청크 + 자동 resume)

데이터 구조 (확인 완료):
```
train_patch (분할 tar, Drive)  ->  train/<task>/{input,target}/*.png          (512x512, 난이도 구분 없음)
val_patch.tar (Drive)          ->  val/<task>/<difficulty>/{input,target}/*    (easy|medium|hard)
test (이미 추출됨)             ->  test/<task>/<difficulty>/input/*            (GT 없음 -> online eval)
tasks: clean dust finger mixed scratch water  /  우리 학습엔 finger dust water scratch 만 사용
```

전체 흐름:
```
[메인 계정] 셀 P 1회 실행  ->  Drive에 SIDL_data/{train.tar, val.tar} 생성 (깔끔/경량)
                          ->  SIDL_data 폴더를 나머지 4계정에 Drive 공유
[각 계정/세션] 셀 1~4 + 셀 D(데이터 로컬 전개) + 셀 R(학습)  ->  2시간마다 셀 R 재실행으로 이어서 학습
```

---

## 셀 P — (메인 계정에서 1회만) 데이터 준비: 분할 tar 합치고 4개 학습타입만 경량 재패키징

```python
from google.colab import drive
drive.mount('/content/drive')

import os, glob, shutil, subprocess
DRIVE = '/content/drive/MyDrive'
os.makedirs('/content/data', exist_ok=True)
os.makedirs(f'{DRIVE}/SIDL_data', exist_ok=True)
TASKS = ['finger', 'dust', 'water', 'scratch']

# 1) train_patch 분할 tar 합치기 (base .tar 가 정렬상 맨 앞 -> 순서 안전)
tp = f'{DRIVE}/train_patch'
parts = sorted(glob.glob(f'{tp}/train_patch.tar*'))
print('합칠 조각:', [os.path.basename(p) for p in parts])
with open('/content/train_full.tar', 'wb') as out:
    for p in parts:
        with open(p, 'rb') as f:
            shutil.copyfileobj(f, out, length=32 * 1024 * 1024)
print('합치기 완료. 내부 경로 확인:')
subprocess.run('tar tf /content/train_full.tar | head -3', shell=True)

# 2) 전체 추출(strip 4 -> train/<task>/...) 후 4개 타입만 재패키징
subprocess.run(['tar', 'xf', '/content/train_full.tar', '-C', '/content/data',
                '--strip-components=4'], check=True)
keep = [f'train/{t}' for t in TASKS]
subprocess.run(['tar', 'cf', '/content/train.tar', '-C', '/content/data'] + keep, check=True)
shutil.move('/content/train.tar', f'{DRIVE}/SIDL_data/train.tar')
os.remove('/content/train_full.tar')

# 3) val_patch -> 그대로 전개 후 재패키징 (모든 task/difficulty 포함)
subprocess.run(['tar', 'xf', f'{DRIVE}/val_patch.tar', '-C', '/content/data',
                '--strip-components=4'], check=True)
subprocess.run(['tar', 'cf', '/content/val.tar', '-C', '/content/data', 'val'], check=True)
shutil.move('/content/val.tar', f'{DRIVE}/SIDL_data/val.tar')

for f in ['train.tar', 'val.tar']:
    print(f'SIDL_data/{f}: {os.path.getsize(f"{DRIVE}/SIDL_data/{f}")/1e9:.2f} GB')
print('\n✅ 완료. 이제 Drive에서 SIDL_data 폴더를 나머지 4계정에 공유하세요.')
```
> 끝나면: Drive 웹에서 `SIDL_data` 우클릭 → 공유 → 나머지 4계정 이메일 추가(뷰어 OK).
> 각 계정은 공유 링크 열어 **"내 드라이브에 바로가기 추가"**.

---

## 셀 1 — Drive 마운트 (세션마다)

```python
from google.colab import drive
drive.mount('/content/drive')
```

## 셀 2 — NAFNet repo 준비 (최초 1회만 clone)

```python
import os
NAFNET = '/content/drive/MyDrive/NAFNet'
if not os.path.isdir(NAFNET):
    %cd /content/drive/MyDrive
    !git clone https://github.com/megvii-research/NAFNet
print('NAFNet ready')
```

## 셀 3 — 의존성 설치 (세션마다)

```python
%cd /content/drive/MyDrive/NAFNet
!pip -q install -r requirements.txt
!pip -q install --upgrade --no-cache-dir gdown calflops
!python3 setup.py develop --no_cuda_ext
```

## 셀 4 — 우리 코드(통합 dataset + config) 주입 (세션마다, 멱등)

```python
REPO_URL = 'https://github.com/gidgogo/deeplearning_project.git'
OURS = '/content/sidl_project'
import os, shutil
if os.path.isdir(OURS): shutil.rmtree(OURS)
!git clone -q {REPO_URL} {OURS}
!cp {OURS}/ext/sidl_multitask_dataset.py /content/drive/MyDrive/NAFNet/basicsr/data/
os.makedirs('/content/drive/MyDrive/sidl_options', exist_ok=True)
!cp {OURS}/configs/*.yml /content/drive/MyDrive/sidl_options/
print('injected dataset + configs')
```

## 셀 D — 데이터를 로컬(/content/data)로 전개 (세션마다, 학습 가속 핵심)

```python
import os, subprocess, time
DRIVE = '/content/drive/MyDrive'
os.makedirs('/content/data', exist_ok=True)
t0 = time.time()
for name in ['train', 'val']:
    if not os.path.isdir(f'/content/data/{name}'):
        subprocess.run(['cp', f'{DRIVE}/SIDL_data/{name}.tar', f'/content/{name}.tar'], check=True)
        subprocess.run(['tar', 'xf', f'/content/{name}.tar', '-C', '/content/data'], check=True)
        os.remove(f'/content/{name}.tar')
print('데이터 준비 완료', round(time.time() - t0), 's')
!echo "train tasks:" && ls /content/data/train && echo "val:" && ls /content/data/val
```

---

## 셀 R — 실험 선택 + resume 자동설정 + 학습 (★ 2시간마다 재실행하는 셀)

```python
EXP = 'A'   # 계정마다 A/B/C/D/E 중 하나

import os, re, glob
TEMPLATES = {
    'A': '/content/drive/MyDrive/sidl_options/SIDL_allinone_A_baseline.yml',
    # B, C, D, E 는 다음 단계에서 추가
}
template_path = TEMPLATES[EXP]
runtime_path = f'/content/drive/MyDrive/sidl_options/_runtime_{EXP}.yml'

txt = open(template_path).read()
name = re.search(r'^name:\s*(\S+)', txt, re.M).group(1)
root = re.search(r'^\s*root:\s*(\S+)', txt, re.M).group(1)
states_dir = os.path.join(root, 'experiments', name, 'training_states')

latest = None
if os.path.isdir(states_dir):
    s = glob.glob(os.path.join(states_dir, '*.state'))
    if s:
        latest = max(s, key=lambda p: int(re.findall(r'(\d+)\.state', p)[0]))
repl = latest if latest else '~'
txt = re.sub(r'(\n\s*resume_state:\s*).*', r'\g<1>' + repl, txt, count=1)
open(runtime_path, 'w').write(txt)
print(f'[{EXP}] name={name}\n  resume_state = {repl}')
```

```python
%cd /content/drive/MyDrive/NAFNet
!torchrun --standalone --nnodes=1 --nproc_per_node=1 --master_port=4311 \
  basicsr/train.py -opt {runtime_path} --launcher pytorch
```

> 세션이 끊기면: 셀 1 → 3 → 4 → D → R 순서로 다시 실행하면 마지막 체크포인트부터 이어서 학습.
> (셀 2는 최초 1회만, 셀 P는 메인 계정 최초 1회만)

---

## 진행 확인 / 산출물
- 체크포인트: `NAFNet_experiments/experiments/<name>/models/net_g_*.pth`
- 학습상태: `.../training_states/*.state`  ← resume 핵심
- 로그/PSNR: `.../train_<name>_*.log`
- `total_iter`(100000) 도달 시 자동 종료.

## 계정 ↔ 실험
| 계정 | EXP |
|------|-----|
| 1 | A baseline |
| 2 | B +aug |
| 3 | C +freq loss |
| 4 | D +deg-cond |
| 5 | E all |
