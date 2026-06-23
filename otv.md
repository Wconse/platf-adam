"ДОЛГ 1 — латеральный разброс аксонных коллатералей (анатомический якорь для σ_EE)
1) “Самый прямой” якорь для локальной рекуррентной E→E в незрелой коре: L4 spiny stellate в barrel cortex
Что даёт: реальный порядок величины для латерального охвата локальных возбуждающих коллатералей (то есть то, что ты хочешь превратить в σ_EE).

Вид/система: крыса (rat), соматосенсорная кора, L4 “barrel” (в статье — “developing rat somatosensory cortex”).
Число: “A spiny stellate cell has a roughly cylindrical axonal field with a cross-section of ∼250 μm in diameter.”
То есть типичный латеральный охват ~250 μm (диаметр), ~125 μm (радиус) для L4 excitatory локальной сети/колонки. 
1
Как это превращать в σ_EE (не как подгонку, а как геометрическую привязку):
если у тебя в коде вероятность связи ~exp(−d²/2σ²), то “характерный радиус” обычно 2–3σ. Тогда σ_EE порядка ~40–80 μm (чтобы 3σ ≈ 120–240 μm) будет согласован с “~250 μm диаметр” как с анатомическим масштабом локального L4 охвата. (Это именно привязка масштаба, а не тюнинг под скорость волн.)

2) “Дальнобойные” горизонтальные коллатерали L2/3: не один масштаб, а (как минимум) два — размер патча и межпатчевое расстояние
Это важно, потому что σ_EE у тебя сейчас пытаются использовать для объяснения и локальных доменов, и глобальной синхронизации/распространения. В реальной коре L2/3 горизонтальные связи обычно кластеризованы.

Вид/система: развивающаяся (developing) ferret/cat visual cortex; “horizontal connections / clustered horizontal inputs”.
Числа (анатомический/функциональный масштаб патчей):
“clustered patches … occupy regions 300–500 μm in diameter” (диаметр патча).
“typical distance 800–1000 μm between clustered horizontal connections observed anatomically in the cat cortex” (межпатчевое расстояние). 
2
Смысл для σ_EE: это уже не “один гаусс”. Если ты хочешь честно моделировать L2/3 горизонтальные связи, то один σ_EE может быть принципиально недостаточен: нужны либо (i) смесь локального + дальнего, либо (ii) структурная пластичность, которая сама кластеризует.

3) Незрелость горизонтальных коллатералей: рост измерим и довольно быстрый (мкм/час), но это не про dw20‑эквивалент L4
Чтобы у тебя был независимый “темп роста” (для Phase‑2 refinement), есть количественные данные по росту горизонтальных аксонов L2/3.

Вид/система: кортикальные L2/3 пирамидные нейроны, developing cortex; time‑lapse с оптогенетической стимуляцией.
Число: средняя скорость роста горизонтальных аксонов 29.9 ± 1.7 μm/час (min 9.5, max 46.7 μm/час). 
3
И отдельный количественный якорь на “типичную длину” горизонтального аксона в organotypic условиях:

Число: средняя длина горизонтальных аксонов в культуре 1540 ± 236 μm. 
4
Зачем это тебе: это позволяет задать независимый масштаб “как быстро может вырасти горизонтальная архитектура”, не выводя его из твоей же скорости волн.

ДОЛГ 2 — “кривая выживания” субплейта по неделям/дням (и почему Allendoerfer & Shatz 1994 не всегда даёт нужный график)
Ты хотел именно “survival curve”. В классике Allendoerfer & Shatz (1994) — обзор (и часто без оцифрованной кривой “% alive vs time”). Но есть первичники, которые дают временной курс плотности/убыли субплейтных/white‑matter нейронов.

1) Прямая кривая (точки по возрастам): SP и WM плотность P2→P30 в коре крысы
Torres‑Reveron & Friedlander (2007, rat visual cortex) реально измеряют убывание плотности “SP neurons” и “WM neurons” на наборе возрастов:

Вид/система: крыса (Sprague Dawley), visual cortex; считали нейроны (NeuN/MAP‑2) в SP и WM.
Возрастные точки (есть кривая): P2, P9, P16, P23, P30. 
5
Числа (точно из текста):
SP: 61% reduction плотности за первые 30 постнатальных дней; к P30 остаётся ~29,000 ± 1,800 neurons/mm³ в SP. 
5
WM: 85% reduction; к P30 остаётся ~500 ± 60 neurons/mm³ в WM. 
5
Как получить именно “кривую” (а не только endpoints): у них это Figure 1B/C (“Plots of neuronal density … during the first month after birth”). Ты можешь WebFetch‑нуть саму фигуру и оцифровать точки (P2/P9/P16/P23/P30). 
5

2) Современная “кривая” для молекулярно определённой субпопуляции SPN + маркеры апоптоза (мышь)
Если тебе нужна именно “survival / death dynamics” не как общая плотность, а как клеточная смерть/апоптоз (с числами), то eLife 2021 даёт очень “ledger‑ready” квантификацию по времени.

Вид/система: мышь, S1 barrel field; Lpar1‑EGFP SPNs; сравнение P3–4 vs P5–6.
Ключевые факты (числа):
нашли только n = 2 / 420 EGFP+/cleaved‑Caspase‑3+ SPNs (5 животных) — то есть Casp‑3 позитивных среди EGFP‑SPN очень мало;
при этом фиксируют значимое снижение плотности EGFP+ SPNs между P3–4 и P5–6 и рост TUNEL‑профилей на P5–6 (и это не строго “только субплейт”, а более широкий слой‑специфичный всплеск смерти). 
6
Это не “весь субплейт”, а чётко определённая под‑популяция; но плюс в том, что это честная “кривая/окно” вокруг P4→P6, где многие модели предполагают реорганизацию.

3) Классический временной якорь “когда начинается смерть SP” в ferret/cat: p75/NGFR исчезает к E60, и дальше “subplate neurons begin to die”
Это не survival‑кривая в процентах, но это очень хороший тайм‑якорь, который обычно используют как “начало окна гибели”.

Вид/система: ferret + cat (в статье прямо: “ferret and cat brain”).
Числа: NGFR/p75 на телах SPNs:
сильное окрашивание уже к E30 и держится ~3 недели,
затем начинает падать около E52,
исчезает к E60, “at which time subplate neurons begin to die.” 
7
Это удобно для твоей “dw‑траектории” (когда включать механизм apoptosis/refinement), но само по себе не даёт %‑кривую.

4) Важная оговорка (чтобы не сделать ложное сильное утверждение): “в грызуне субплейт обязателен к массовой гибели” — оспаривается
Есть сильный первичник (J Neurosci 1995), который прямо проверял “универсальна ли смерть субплейта в rodent” и получил отрицательный результат на их определении “rodent subplate = layer VIb”:

Вид/система: крыса; слой VIb; birthdating E12–E15, анализ P1→P63; дополнительно pyknotic/TUNEL‑подобные оценки E18→P15.
Результат: число “heavily labeled” раннерождённых клеток в VIb остаётся постоянным (ANOVA: незначимые различия) от P1 до P63; и уровень cell death в VIb не выделяется на фоне остальных слоёв (кроме layer II и белого вещества). 
8
Почему это важно: “survival curve” зависит от того, что ты называешь субплейтом (VIb vs transient SP proper vs WM interstitial), и от метода (плотность NeuN‑клеток vs birthdated cohort). Это не отменяет Torres‑Reveron (там про SP/WM плотность), но заставляет аккуратно формулировать “кто именно умирает”.

URL (как просил — ссылками в конце, чтобы сразу WebFetch)
Латеральный разброс / σ_EE
text

Feldmeyer et al. 1999 (L4 spiny stellate axonal field ~250 µm diameter; developing rat barrel cortex):
https://pmc.ncbi.nlm.nih.gov/articles/PMC2269646/   (если у тебя откроется без CAPTCHA; иначе см. цитируемый фрагмент из поиска)

Functional mapping / clustered horizontal connections (patch 300–500 µm; spacing 800–1000 µm):
https://www.researchgate.net/publication/15203418_Functional_mapping_of_horizontal_connections_in_developing_ferret_visual_cortex_Experiments_and_modeling

Horizontal axon growth rate (29.9±1.7 µm/h):
https://pmc.ncbi.nlm.nih.gov/articles/PMC3871609/

Horizontal axon length in culture (1540±236 µm; plus “no branches at P7” claim inside):
https://pmc.ncbi.nlm.nih.gov/articles/PMC6725200/
SP survival / death dynamics
text

Torres-Reveron & Friedlander 2007 (SP & WM density P2/P9/P16/P23/P30; 61% SP drop; P30 ~29k neurons/mm³; Figure 1 density curves):
https://pmc.ncbi.nlm.nih.gov/articles/PMC6672634/
Figure image (direct CDN; удобно для оцифровки):
https://cdn.ncbi.nlm.nih.gov/pmc/blobs/067d/6672634/8789753caf23/zns0350737630001.jpg

eLife 2021 (Lpar1-EGFP SPNs; P3–4 vs P5–6; n=2/420 EGFP+/Casp-3+):
https://cdn.elifesciences.org/articles/60810/elife-60810-v3.pdf

Allendoerfer et al. 1990 PNAS (NGFR/p75 timeline: E30 strong ~3 weeks; declines ~E52; gone by E60; then SPNs begin to die):
https://pmc.ncbi.nlm.nih.gov/articles/PMC53226/   (если PMC не даёт из-за CAPTCHA, этот же abstract есть в поисковом сниппете)

Valverde et al. 1995 J Neurosci (rodent VIb: early-born cohort count constant P1–P63; death in VIb not prominent):
https://www.researchgate.net/publication/15576277_Persistence_of_early-generated_neurons_in_the_rodent_subplate_Assessment_of_cell_death_in_neocortex_during_the_early_postnatal_period"