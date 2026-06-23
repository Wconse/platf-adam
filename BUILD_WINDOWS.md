# Сборка cuda_adam на Windows

## Окружение

| Что | Путь |
|-----|------|
| Visual Studio (MSVC 14.51) | `E:\VIS STUD\` |
| Windows SDK 11 (10.0.26100) | `E:\Windows Kits\10\` |
| CUDA Toolkit 13.3 | `D:\CUDAtoolkit\` |
| GPU | RTX 30xx, sm_86 |
| Shell | MSYS2 / MinGW64 (bash) |
| CMake | 4.4.0 (`C:\Program Files\CMake\`) |
| Ninja | из Python Scripts |

> Все пути нестандартные — Program Files перенесены через junction-точки на D: и E: для экономия места на C:.

---

## Быстрый старт

### Из cmd (полная сборка + запуск)
```bat
build.bat
```
Brian2-шаги пропускаются автоматически если brian2 не установлен, существующие ref-файлы используются как есть.

### Только симуляция (без пересборки)
```bat
cuda\build\sim_runner.exe reference output
```

### Только мультирегион
```bat
cuda\build\sim_multiregion.exe <bundle_dir> <output_dir>
```

---

## Ручная сборка из bash (MSYS2)

Нужно явно прокинуть MSVC и SDK в окружение — они не в системном PATH:

```bash
MSVC_BIN="/e/VIS STUD/VC/Tools/MSVC/14.51.36231/bin/Hostx64/x64"
SDK_BIN="/e/Windows Kits/10/bin/10.0.26100.0/x64"
SDK_LIB_UM="E:/Windows Kits/10/Lib/10.0.26100.0/um/x64"
SDK_LIB_UCRT="E:/Windows Kits/10/Lib/10.0.26100.0/ucrt/x64"
MSVC_LIB="E:/VIS STUD/VC/Tools/MSVC/14.51.36231/lib/x64"

rm -rf cuda/build && mkdir cuda/build && cd cuda/build

PATH="$MSVC_BIN:$SDK_BIN:$PATH" \
LIB="$SDK_LIB_UM;$SDK_LIB_UCRT;$MSVC_LIB" \
cmake .. -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DCMAKE_CUDA_HOST_COMPILER="E:/VIS STUD/VC/Tools/MSVC/14.51.36231/bin/Hostx64/x64/cl.exe" \
  -DCMAKE_EXE_LINKER_FLAGS="/MANIFEST:NO"

PATH="$MSVC_BIN:$SDK_BIN:$PATH" \
LIB="$SDK_LIB_UM;$SDK_LIB_UCRT;$MSVC_LIB" \
cmake --build . --parallel
```

---

## Грабли, на которые уже наступили

### 1. `nvcc: Cannot find compiler 'cl.exe' in PATH`
nvcc на Windows **всегда** ищет `cl.exe` по имени в PATH — cmake-флаги тут не помогают.  
Решение: добавить `E:\VIS STUD\VC\Tools\MSVC\14.51.36231\bin\Hostx64\x64` в PATH перед запуском cmake.

### 2. Git's `link.exe` перехватывает MSVC `link.exe`
Git устанавливает свой `link.exe` (`C:\Program Files\Git\usr\bin\link.exe`), который несовместим с MSVC-линковкой.  
Решение: ставить MSVC bin **раньше** Git в PATH: `PATH="$MSVC_BIN:$PATH"`.

### 3. `LNK1104: не удается открыть файл "kernel32.lib"`
Линковщик не находит Windows SDK библиотеки.  
Причина: SDK установлен в нестандартном месте (`E:\Windows Kits\`), переменная `LIB` не выставлена.  
Решение: явно передавать `LIB="$SDK_LIB_UM;$SDK_LIB_UCRT;$MSVC_LIB"` при вызове cmake.

### 4. `D8016: несовместимые параметры /RTC1 и /O2`
cmake без явного build type создаёт Debug-сборку с `/RTC1 /Od`, что конфликтует с `-O2` в CMakeLists.txt.  
**Важно:** с Ninja генератором `cmake --build . --config Release` не работает (Ninja single-config).  
Решение: передавать тип при конфигурации: `-DCMAKE_BUILD_TYPE=Release`.

### 5. `LNK2019` для sim_multiregion — unresolved externals
Симптом: линковщик жалуется на несовпадение сигнатур функций вида:  
```
ожидается: accumulate_spike_counts(int const *, int *, int)
найдено:   accumulate_spike_counts(int const * __restrict, int * __restrict, int)
```
Причина: на MSVC `__restrict__` влияет на C++ name mangling. Форвард-декларации в `main_multiregion.cu` не имели `__restrict__`, а определения в kernel-файлах имели.  
Решение: добавить `__restrict__` во все форвард-декларации в `main_multiregion.cu` (~строки 154–272).  
Затронутые функции: `accumulate_spike_counts`, `afferent_poisson_step`, `init_afferent_curand`, `stdp_pre_pathway`, `stdp_clear_ring_slot`, `stdp_post_pathway`, `stdp_enqueue_mark`.

### 6. VS Installer — где лежит
```
C:\Program Files (x86)\Microsoft Visual Studio\Installer\vs_installer.exe
```
Из bash запускать через cmd: `cmd /c "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vs_installer.exe"`

---

## Структура build-артефактов

```
cuda/build/
  sim_runner.exe        # M1/M2: одиночный регион, pre-generated noise
  sim_multiregion.exe   # M3+: мультирегион, cuRAND noise, STDP
  *.pdb                 # debug symbols (линковались с /debug даже в Release — ок)
```

---

## Если нужно обновить MSVC версию

Путь `14.51.36231` — конкретная версия тулсета. При обновлении VS папка изменится.  
Найти актуальную:
```bash
ls "E:/VIS STUD/VC/Tools/MSVC/"
```
Обновить в `build.bat` и в ручной команде выше.

## Если нужно обновить Windows SDK версию

```bash
ls "E:/Windows Kits/10/Lib/"       # покажет установленные версии
ls "E:/Windows Kits/10/bin/"
```
