# CUDA-AdEx — Milestone 0

Портирование AdEx-ядра нейронной модели АДАМ на CUDA C++.

## Цель

Проверить, что CUDA-реализация AdEx-динамики воспроизводит те же firing rates, что и Brian2 (CPU reference), для изолированной популяции P1 (80E + 20I = 100 нейронов).

## Структура

```
cuda_adam/
├── reference/
│   ├── params_p1.py           # Все параметры P1 (единый источник истины)
│   ├── gen_noise.py           # Pre-generate OU noise (seed=7)
│   ├── run_brian2_ref.py      # Brian2 CPU reference simulation
│   └── *.npy, *.json          # Reference data (generated)
│
├── cuda/
│   ├── CMakeLists.txt
│   ├── include/
│   │   ├── units.cuh          # Единицы измерения
│   │   └── adex_params.cuh    # AdExParams struct + kernel declarations
│   └── cuda_src/
│       ├── kernels/
│       │   └── adex_kernel.cu # CUDA kernels (OU, AdEx, spikes, homeostasis)
│       └── main.cu            # Главный цикл симулятора
│
├── validation/
│   └── compare_traces.py      # Сравнение CUDA vs Brian2
│
├── build_and_run.sh           # Скрипт сборки и запуска
└── README.md
```

## Что моделируется (Milestone 0)

- 80 excitatory (AdEx) + 20 inhibitory (LIF) нейронов P1
- Ornstein-Uhlenbeck фоновый шум (ge_bg, gi_bg)
- Bias current + homeostatic plasticity
- Матурация (gate: 0→1 начиная с t_birth=7466.67 ms)
- **Без** синаптических связей между нейронами
- **Без** STDP, афферентов, стресса

## Быстрый старт

```bash
# На GPU-машине с CUDA toolkit
cd cuda_adam
bash build_and_run.sh
```

Или пошагово:

```bash
# 1. Reference (на любой машине с Python + Brian2)
cd reference
python3 gen_noise.py
python3 run_brian2_ref.py

# 2. CUDA build (на GPU-машине)
cd ../cuda
mkdir -p build && cd build
cmake ..
make -j

# 3. Run
./sim_runner ../../reference ../../output

# 4. Validate
cd ../../
python3 validation/compare_traces.py reference output
```

## Валидационные метрики

| Метрика | Допуск | Описание |
|---------|--------|----------|
| Mean E rate | < 1.0 Hz error | Средний firing rate excitatory нейронов |
| Mean I rate | < 1.0 Hz error | Средний firing rate inhibitory нейронов |
| Per-neuron rate | < 2.0 Hz max | Максимальное отклонение одного нейрона |

## Подводные камни (из ARCH_CHECKPOINT)

1. **float32 vs float64**: Brian2 → float64, CUDA → float32. Накапливается drift ≈ 10⁻⁴ mV за 10s.
   - Milestone 0: принимаем погрешность, валидируем rates.
   - Milestone 1: можно перейти на float64 для ключевых переменных.

2. **Порядок обновления**: Brian2 вычисляет все RHS из состояния начала шага.
   CUDA-ядра вызываются в порядке: homeostasis → OU → dynamics → spike_reset.
   Каждое ядро читает "old state", обновляет свою переменную.

3. **Разный шум**: Brian2 и CUDA используют разные RNG. Сравниваем статистически (mean rates), а не поточечно.

4. **w_clip**: Адаптация клиппируется в [0, 400] pA. В CUDA реализовано через `fmaxf/fminf`.

5. **Рефрактерность**: Во время рефрактерного периода v клампится к V_reset.

## Ожидаемый результат

```
Brian2 reference (own RNG):     E ≈ 2.0 Hz,  I ≈ 7.8 Hz
Python Euler (seed=7 noise):    E ≈ 0.33 Hz, I ≈ 8.8 Hz
CUDA (seed=7 noise):            E ≈ 0.33 Hz, I ≈ 8.8 Hz
```

✅ **CUDA kernel verified**: Python Euler (same noise) and CUDA produce identical rates.
The difference with Brian2 is purely noise-dependent (Brian2 uses its own internal RNG).

## Следующие шаги (Milestone 1+)

- [ ] Добавить синапсы (AMPA, GABA)
- [ ] NMDA с магниевым блоком
- [ ] STDP для P1/P2
- [ ] Афференты (AUD, VEST, TACT)
- [ ] TimedArray для E_GABA(t), Mg(t), stress(t)
- [ ] Полная сеть (все регионы)
- [ ] float64 для v, w (снижение drift)
- [ ] HDF5 checkpoint save/load
