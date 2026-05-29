# Colab 실행 가이드 (2시간 청크 + 자동 resume)

> 5개 계정에서 **이 노트북 1개를 공유**해서 씁니다.
> 맨 위 `EXP` 변수만 `A/B/C/D/E`로 바꾸면 그 실험이 돌아갑니다.
> 세션이 끊기거나 2시간이 지나면 **마지막 셀만 다시 실행** → 자동으로 이어서 학습.

각 계정 사전 준비(1회):
1. Colab에서 **런타임 → 런타임 유형 변경 → GPU(T4)** 설정
2. SIDL 공유 폴더 3개(Train/Val/Test)를 **내 드라이브에 바로가기 추가**
3. (선택) 우리 코드 GitHub repo URL 준비 → `REPO_URL`에 입력

---

## 셀 1 — Drive 마운트 + 데이터 구조 검증 (★ 제일 먼저)

```python
from google.colab import drive
drive.mount('/content/drive')

import os
# ↓ 마운트 후 실제 폴더명에 맞게 수정 (Train/Validation/Test 가 어떻게 보이는지 확인)
TRAIN_ROOT = '/content/drive/MyDrive/train'
VAL_ROOT   = '/content/drive/MyDrive/val'
TEST_ROOT  = '/content/drive/MyDrive/test'

def peek(root):
    print(f'\n=== {root} ===  exists={os.path.isdir(root)}')
    if os.path.isdir(root):
        for t in sorted(os.listdir(root))[:8]:
            p = os.path.join(root, t)
            sub = os.listdir(p)[:6] if os.path.isdir(p) else p
            print(f'  {t}/ -> {sub}')

for r in (TRAIN_ROOT, VAL_ROOT, TEST_ROOT):
    peek(r)

# 기대 구조:
#   train/<task>/input , train/<task>/target           (task: finger dust water scratch ...)
#   val/<task>/<difficulty>/input|target               (difficulty: easy medium hard)
#   test/<task>/<difficulty>/input                      (GT 없음)
```
**→ 이 셀 출력을 보고 실제 폴더명/경로가 위 기대와 다르면 알려주세요. config의 dataroot를 맞춥니다.**

---

## 셀 2 — NAFNet repo 준비 (최초 1회만 git clone)

```python
NAFNET = '/content/drive/MyDrive/NAFNet'
import os
if not os.path.isdir(NAFNET):
    %cd /content/drive/MyDrive
    !git clone https://github.com/megvii-research/NAFNet
print('NAFNet ready at', NAFNET)
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
REPO_URL = ''   # ← 우리 GitHub repo (예: https://github.com/<you>/sidl-project)
OURS = '/content/sidl_project'

import os, shutil
if REPO_URL:
    if os.path.isdir(OURS): shutil.rmtree(OURS)
    !git clone {REPO_URL} {OURS}
    # 통합 dataset을 basicsr/data 로 복사 → *_dataset.py 자동 등록
    !cp {OURS}/ext/sidl_multitask_dataset.py /content/drive/MyDrive/NAFNet/basicsr/data/
    # config들을 작업 폴더로 복사
    os.makedirs('/content/drive/MyDrive/sidl_options', exist_ok=True)
    !cp {OURS}/configs/*.yml /content/drive/MyDrive/sidl_options/
    print('injected dataset + configs')
else:
    print('REPO_URL 비어있음 — 아래 fallback 셀로 파일 직접 작성하세요')
```

> GitHub 안 쓰면: `sidl_multitask_dataset.py`를 `NAFNet/basicsr/data/`에,
> config yml을 `/content/drive/MyDrive/sidl_options/`에 직접 업로드해도 됩니다.

---

## 셀 5 — (권장) 데이터를 로컬로 복사해 I/O 가속

> Drive에서 직접 읽으면 매우 느립니다. 로컬 /content로 옮기면 학습이 몇 배 빨라집니다.
> 처음엔 생략하고 파이프라인부터 검증해도 됩니다.

```python
# 4개 학습 타입 + medium val 만 로컬 복사 (필요한 것만)
import os, shutil, time
LOCAL_TRAIN = '/content/train'; LOCAL_VAL = '/content/val'
TASKS = ['finger', 'dust', 'water', 'scratch']
t0 = time.time()
for t in TASKS:
    for split_src, split_dst in [(TRAIN_ROOT, LOCAL_TRAIN)]:
        src = os.path.join(split_src, t); dst = os.path.join(split_dst, t)
        if os.path.isdir(src) and not os.path.isdir(dst):
            shutil.copytree(src, dst)
    vsrc = os.path.join(VAL_ROOT, t, 'medium'); vdst = os.path.join(LOCAL_VAL, t, 'medium')
    if os.path.isdir(vsrc) and not os.path.isdir(vdst):
        os.makedirs(os.path.dirname(vdst), exist_ok=True); shutil.copytree(vsrc, vdst)
print('copied in', round(time.time()-t0), 's')
# 로컬을 쓰려면 config dataroot를 /content/train, /content/val 로 바꿔야 함 (셀 6에서 처리)
USE_LOCAL = True
```

---

## 셀 6 — 실험 선택 + resume_state 자동 설정 → 학습 시작 (★ 반복 실행 셀)

```python
EXP = 'A'   # ← A / B / C / D / E 중 선택 (계정마다 다르게)

import os, re, glob

TEMPLATES = {
    'A': '/content/drive/MyDrive/sidl_options/SIDL_allinone_A_baseline.yml',
    # B,C,D,E 는 이후 단계에서 추가
}
template_path = TEMPLATES[EXP]
runtime_path  = f'/content/drive/MyDrive/sidl_options/_runtime_{EXP}.yml'

txt = open(template_path).read()

# (옵션) 로컬 데이터 사용 시 dataroot 치환
if 'USE_LOCAL' in dir() and USE_LOCAL:
    txt = txt.replace('dataroot: /content/drive/MyDrive/train', 'dataroot: /content/train')
    txt = txt.replace('dataroot: /content/drive/MyDrive/val',   'dataroot: /content/val')

name = re.search(r'^name:\s*(\S+)', txt, re.M).group(1)
root = re.search(r'^\s*root:\s*(\S+)', txt, re.M).group(1)
states_dir = os.path.join(root, 'experiments', name, 'training_states')

latest = None
if os.path.isdir(states_dir):
    states = glob.glob(os.path.join(states_dir, '*.state'))
    if states:
        latest = max(states, key=lambda p: int(re.findall(r'(\d+)\.state', p)[0]))
repl = latest if latest else '~'
# resume_state 줄만 치환 (!!float 등 나머지 보존)
txt = re.sub(r'(\n\s*resume_state:\s*).*', r'\g<1>' + repl, txt, count=1)
open(runtime_path, 'w').write(txt)
print(f'[{EXP}] name={name}\n  resume_state = {repl}')
```

```python
# 학습 실행 (이 셀이 끊기면 위 셀 + 이 셀만 다시 실행하면 이어서 진행됨)
%cd /content/drive/MyDrive/NAFNet
!torchrun --standalone --nnodes=1 --nproc_per_node=1 --master_port=4311 \
  basicsr/train.py -opt {runtime_path} --launcher pytorch
```

---

## 진행 확인
- 체크포인트: `NAFNet_experiments/experiments/<name>/models/net_g_*.pth`
- 학습 상태: `.../training_states/*.state`  ← resume의 핵심
- 로그/PSNR: `.../train_<name>_*.log`
- `total_iter`(100000) 도달하면 자동 종료.

## 계정 ↔ 실험 배정
| 계정 | EXP |
|------|-----|
| 1 | A (baseline) |
| 2 | B (+aug) |
| 3 | C (+freq loss) |
| 4 | D (+deg-cond) |
| 5 | E (all) |
