# Plan: de cero a una IA campeona de dominó cubano 🁫

Meta: aprender Reinforcement Learning construyendo un agente que juegue
**dominó cubano de 9** (doble-9, 55 fichas, parejas 2v2) a gran nivel.
Restricción: **sin GPU potente** — solo CPU y las ~30h gratis de Colab.

Buenas noticias primero: **no necesitas GPU para esto.** La red que vamos a
entrenar es diminuta (143 → 128 → 128 → 111 neuronas) y el simulador es
Python puro y rapidísimo. El cuello de botella en RL de juegos no es el
hardware, es el diseño: cómo representas el estado, cómo defines la
recompensa y contra quién entrena el agente. Todo eso corre perfecto en CPU.

## Principios (no negociables)

1. **Bitter lesson estricta.** Cero heurísticas humanas inyectadas en el
   aprendizaje: ni reglas de "juega el doble", ni conteo de fichas
   programado, ni recompensas que premien "buenas jugadas" según un humano.
   La única señal es ganar o perder (+1/−1). Todo lo demás — contar fichas,
   inferir manos ocultas, coordinarse con el compañero sin hablarle — debe
   **emerger** del self-play o no existe.
2. **Doble propósito.** (a) Un agente que juegue a gran nivel. (b) Usar el
   agente entrenado como **herramienta de descubrimiento**: extraer las
   heurísticas que infirió solo y devolvérselas a los humanos en forma
   legible (Fase 7). RL como microscopio estadístico, no solo como
   marcador.
3. **El juego es un Dec-POMDP**, no un MDP: 4 agentes, información parcial
   (cada uno ve solo su mano), recompensa de equipo, y coordinación con el
   compañero **sin canal de comunicación** — cualquier "señal" debe emerger
   de las jugadas mismas (misma familia que el bidding en Bridge o Hanabi).

## El mapa mental (RL para quien viene de supervised learning)

| Supervised learning | Reinforcement learning |
|---|---|
| Dataset etiquetado (X, y) | Simulador que genera partidas infinitas |
| Loss que minimizas | Recompensa que maximizas (+1 ganar / −1 perder) |
| Épocas sobre datos fijos | Episodios jugando contra sí mismo (*self-play*) |
| El modelo predice | La **política** decide qué jugada hacer |
| Overfitting a datos | Overfitting a un rival débil (por eso hay torneo) |

Vocabulario mínimo: **agente** (el que decide), **entorno** (el juego),
**estado** (lo que ve el agente), **acción** (la jugada), **recompensa**,
**política π** (la estrategia: estado → acción), **valor V/Q** (cuánto vale
un estado/jugada), **exploración vs explotación** (probar jugadas nuevas vs
repetir las que funcionan), **γ (descuento)**: cuánto importan las
recompensas futuras.

---

## Fase 0 — Fundamentos (2–4 días, 1h/día)

**Objetivo:** entender el loop de RL sin código todavía.

1. Lee los capítulos 1–3 del libro *Reinforcement Learning: An Introduction*
   (Sutton & Barto, gratis en internet). Solo la intuición, no las mates.
2. Alternativa/complemento en video: el curso gratuito *Deep RL Course* de
   Hugging Face (corto, práctico, con código).
3. Entiende estos 3 algoritmos a nivel "qué idea tienen":
   - **Q-learning**: aprender una tabla "estado → valor de cada acción".
   - **DQN**: Q-learning pero la tabla es una red neuronal.
   - **PPO**: en vez de valores, ajusta la política directamente, con pasos
     conservadores para no "romper" lo aprendido. Es el que usaremos.

**Listo cuando:** puedas explicar con tus palabras qué es un episodio, una
recompensa y por qué el agente necesita explorar.

## Fase 1 — El simulador ✅ (hecha)

**Objetivo:** convertir el dominó en un entorno con el que una IA pueda
interactuar millones de veces.

Ya implementado en `src/domino_rl/game.py`: reparto, salida obligada con el
doble más alto, jugadas legales, paso, dominó, tranca y conteo de tantos.
Verificado con 9 tests (`tests/test_game.py`, escritos *antes* que el motor).

**Qué aprender aquí:** en RL el simulador ES el dataset. Debe ser (a) correcto
en reglas, (b) rapidísimo (millones de partidas), (c) con una API estándar
(`reset`/`step`). Por eso existe `env.py` con interfaz estilo Gymnasium.

**Listo cuando:** `pytest tests/` pasa y un torneo random-vs-random termina
sin errores (ya lo hace).

## Fase 2 — Baselines ✅ (hecha)

**Objetivo:** tener rivales tontos contra los que medir progreso.

Ya implementado en `baselines.py`: `RandomAgent` (juega al azar) y
`GreedyAgent` (heurístico: siempre bota la ficha más alta). Y `evaluate.py`
corre torneos:

```bash
python -m domino_rl.evaluate --rounds 200 --a greedy --b random
```

**Qué aprender aquí:** la métrica que manda es **win-rate contra baselines**,
no la "loss" del entrenamiento. Un agente que no le gana al azar no aprendió
nada, por muy bonita que se vea su curva de reward.

**Listo cuando:** greedy le gana a random de forma consistente (ya: ~53%, con
más puntos por victoria — el heurístico ingenuo apenas supera al azar, lo que
muestra que el dominó no se gana solo "botando lo alto").

## Fase 3 — La representación (1 día)

**Objetivo:** entender *qué ve* el agente y *qué puede hacer*.

- **Observación** (`env.py::observe`): vector de 143 números — mi mano (55),
  puntas de la mesa (20), **historial público de fichas jugadas** (55),
  fichas restantes por jugador (4), pases seguidos (1), quién soy y quién es
  mi pareja (8). Todo normalizado a [0,1]. La tesis: el sistema es cerrado y
  contable (55 fichas, se sabe cuántas salieron), así que el conteo de fichas
  y la inferencia de manos ocultas deben **emerger solos** del historial —
  sin una línea de código que cuente.
- **Acciones** (111): 55 fichas × 2 puntas + pasar. Con **máscara de acciones
  legales**: la red nunca elige una jugada ilegal (truco estándar que acelera
  muchísimo el aprendizaje).
- **Recompensa**: `sparse` (+1/−1/0 al final, **default — bitter lesson**) o
  `shaped` (experimental: empujoncito denso por dejar menos pips; solo para
  medir si acelera sin corromper).

**Qué aprender aquí:** el 80% del éxito en RL aplicado está en estos tres
diseños, no en el algoritmo. Si el agente no aprende, casi siempre el problema
está aquí.

**Listo cuando:** puedas explicar por qué cada bloque de 143 números está ahí.

## Fase 4 — Primer agente: Q-learning tabular (2–3 días) ⭐ tu "hola mundo"

**Objetivo:** *ver* aprender a una tabla antes de usar redes neuronales.

El dominó de 9 completo es muy grande para una tabla, así que lo
simplificamos: **dominó de 6, 2 jugadores, sin parejas** (28 fichas, estados
manejables con una tabla + exploración ε-greedy).

1. Crea `src/domino_rl/tabular.py`: un `defaultdict` como tabla Q,
   actualización `Q(s,a) += α [r + γ·max Q(s′,a′) − Q(s,a)]`.
2. Entrena 200k manos contra `RandomAgent`. Grafica win-rate cada 10k manos.
3. Deberías ver la curva subir de ~50% a >70%: **eso es RL funcionando**,
   sin una sola neurona.

**Qué aprender aquí:** la ecuación de Bellman en la práctica, ε-greedy
(exploración), y por qué las tablas no escalan (aquí nace la necesidad de DQN).

**Listo cuando:** tu tabla le gana al azar >65% en dominó de 6.

## Fase 5 — Deep RL + self-play: el entrenamiento de verdad (1–2 semanas)

**Inspiración**: [DominAI](https://github.com/igorbispo99/dominai) (doble-6,
DQN self-play, modo parejas, máscara de acciones, *opponent mix*) y
[DouZero](https://github.com/kwai/DouZero) (ICML 2021: juego de "shedding" con
colaboración + información incompleta + acciones masivas, resuelto con self-play
deep RL — la prueba de que esta familia de técnicas funciona para nuestra clase
de problema).

**Objetivo:** PPO con red neuronal jugando contra sí mismo en doble-9,
**camino progresivo** (pipeline simple primero, objetivo real después):

1. **1v1 con 2 jugadores** (`--n-players 2 --mode individual`): el problema
   más simple con información oculta real. Solo valida que el pipeline
   aprende (win-rate vs random > 60%).
2. **4 jugadores individual** (`--mode individual`): todos contra todos.
   Valida que el aprendizaje escala a 4 manos ocultas.
3. **2v2 parejas** (`--mode teams`, el objetivo real): coordinación ciega
   con el compañero. Aquí es donde debe emerger la "comunicación" por
   jugadas.

Ya tienes el script: `src/domino_rl/train.py` (usa `MaskablePPO` de
sb3-contrib). El truco clave ya está implementado: **self-play con política
compartida** — el entorno sienta al mismo agente en las sillas que toquen y
cada recompensa se calcula desde la perspectiva del equipo que actúa.
Mezcla de oponentes (checkpoints viejos + bots) cuando el self-play puro se
estanque.

```bash
# En Colab (gratis, CPU): clona el repo, pip install -e ".[train]" y:
python -m domino_rl.train --n-players 2 --mode individual --timesteps 1000000 --n-envs 4 --out /content/drive/MyDrive/domino-rl
```

- 2M de pasos ≈ unas horas en CPU de Colab. Guarda checkpoints (el free
  tier puede desconectarse).
- Hiperparámetros de partida ya puestos: red 128×128, `gamma=0.995`
  (las manos son cortas: lo que importa es el resultado), lr 3e-4.
- Si la curva de win-rate vs greedy no sube en 500k pasos: revisa Fase 3
  (casi siempre es la recompensa o la observación, no los hiperparámetros).

**Qué aprender aquí:** PPO por dentro (a alto nivel), por qué el self-play
evita el overfitting a un rival fijo, y a leer curvas de entrenamiento sin
autoengañarte.

**Listo cuando:** le gana a `greedy` >60% en 500 manos (ver Fase 6).

## Fase 6 — Evaluación rigurosa (2–3 días)

**Objetivo:** saber de verdad qué tan bueno es, y *dónde* pierde.

1. Torneo largo: `evaluate.py --a ppo:checkpoints/final.zip --b greedy
   --rounds 1000`. Reporta win-rate con intervalo de confianza.
2. **Análisis de errores**: guarda 50 manos perdidas y míralas. ¿Pierde por
   mala salida? ¿No defiende? ¿No ayuda a la pareja? Cada patrón → una mejora
   (Fase 7).
3. Mide también: % de dominó vs tranca, tantos promedio por victoria.

**Qué aprender aquí:** evaluar agentes es una disciplina propia. Un solo
número miente; los patrones de derrota enseñan.

## Fase 7 — Descubrimiento de conocimiento (el segundo propósito) ⭐

**Objetivo:** usar al agente entrenado como microscopio estadístico —
extraer las heurísticas que infirió solo y devolvérselas a los humanos en
forma legible. Si la bitter lesson funcionó, aquí aparece la recompensa.

1. **Análisis de política**: muestrea millones de decisiones del agente y
   mide patrones: ¿con qué frecuencia "se dobla" con cada doble según el
   estado de la mesa? ¿Cuándo sacrifica puntas? ¿Cómo cambia su juego según
   lo que ya salió (usa de verdad el historial)?
2. **Situaciones prototípicas**: agrupa estados similares y extrae la jugada
   típica del agente en cada uno → "en mesa X con mano Y, el agente hace Z
   el 87% de las veces".
3. **Ablaciones**: ¿qué pasa si le quitas el historial de la observación?
   ¿Si le quitas la identidad de la pareja? Lo que degrade el rendimiento
   revela qué información *realmente* usa (p. ej. "sin historial pierde 12
   puntos de win-rate → sí aprendió a contar fichas").
4. **Reporte legible**: escribe `docs/hallazgos.md` con las heurísticas
   descubiertas en lenguaje de jugador de dominó, cada una con su evidencia
   estadística. Ejemplo del formato buscado: *"Con la mesa trabada y 3+
   fichas del palo dominante fuera, el agente evita doblarse con dobles
   bajos (p < 0.01, n = 40k situaciones)"*.

**Listo cuando:** un jugador humano puede leer `docs/hallazgos.md` y
aprender al menos 3 ideas que no sabía.

## Fase 8 — Subir de nivel (iterativo, el resto del proyecto)

Ideas ordenadas por impacto esperado. Ataca la que revele la Fase 6:

1. **League play**: entrena contra un *pool* de versiones pasadas del agente
   (no solo contra sí mismo actual) → evita estrategias cíclicas.
2. **Señales a la pareja**: el agente ya ve quién es su pareja; analiza
   (Fase 7) si emergieron jugadas "señal" y refuérzalas con currículo.
3. **MCTS en inferencia**: en tu turno, simula N futuros con el modelo actual
   y elige la mejor jugada (estilo AlphaZero-lite, solo CPU).
4. **Deep CFR**: si el self-play se estanca o quieres rigor teórico, esta es
   la vía académica (más compleja; ojo: CFR vanilla es para 2 jugadores
   suma-cero — el 2v2 por equipos necesita extensiones).

## Fase 9 — Jugar contra ella (1 semana, la diversión) — prototipo ✅

**Objetivo:** una mesa digital donde humanos e IAs juegan juntos.

Ya existe el prototipo en `demo/`: `server.py` (la mesa: motor real +
bots + la red PPO) e `index.html` (cada jugador abre la página en su
teléfono y ve solo su mano). Para probarlo hoy:

```bash
PYTHONPATH=src python demo/server.py   # en tu laptop
# cada jugador abre http://<ip-de-la-laptop>:8000 en su telefono
```

- El Humano 1 crea la mesa (elige modo y rival); el Humano 2 se une.
- En parejas 2v2 los humanos son pareja contra dos IAs; las sillas
  vacías las ocupa la IA rival elegida (greedy = la más fuerte hoy,
  random, o la PPO estudiante cuando exista `demo/models/ppo_smoke.zip`).
- Partido a 100 tantos, la página refresca sola cada 1.5s.

Lo que falta para producción: autenticación/salas múltiples (hoy una sola
mesa por servidor), y cambiar `greedy` por `final.zip` cuando el
entrenamiento real termine — es cambiar una línea.

## Fase 10 — Pareja de IAs vs pareja de humanos (la prueba de fuego) ⭐

**Objetivo:** dos copias del mismo modelo (una por silla, cada una viendo
*solo* su mano) juegan como pareja contra dos humanos.

- Es el experimento que valida todo el proyecto: si la coordinación ciega
  emergió de verdad en el entrenamiento, la pareja de IAs debe coordinarse
  sin hablarse — igual que hacía contra sí misma.
- Implementación: cargar `final.zip` dos veces; cada copia recibe solo la
  observación de su silla (los mismos 143 números desde su perspectiva).
  Es ~30 líneas: un `PPOAgent(round, seat)` que envuelve `model.predict`.
  (En entrenamiento ya hacemos esto mismo con 4 sillas.)
- Interfaz: extender la web de la Fase 9 a 4 sillas (2 humanas + 2 IA), o un
  bot de WhatsApp por turnos.
- Métrica: win-rate de la pareja IA vs parejas humanas en N manos, más la
  pregunta cualitativa: *"¿sentiste que las dos IAs se coordinaban?"*.

---

## Decisiones de diseño ya tomadas (y por qué)

- **Episodio = una mano**, no un partido a 100: episodios cortos = más
  señal de aprendizaje por hora. Los tantos del partido se añaden después.
- **Currículo progresivo**: 1v1 (2 jugadores) → 4 individual → 2v2 parejas.
  El self-play con política compartida maneja todos sin complejidad extra.
- **Bitter lesson**: reward `sparse` por defecto; `shaped` solo como
  experimento controlado.
- **Sin GPU por diseño**: si algún día quieres escalar, el mismo código corre
  en GPU sin cambios (PyTorch lo hace solo).
- **Reglas simplificadas a propósito**: sin capicúa/bonificaciones raras.
  Se añaden cuando el agente domine lo básico.

## Glosario rápido

- **Episodio**: una mano completa, del reparto al conteo de tantos.
- **Política (π)**: la red neuronal; entra el estado (143 números), sale la
  jugada.
- **Self-play**: el agente es su propio rival en las 4 sillas.
- **Action masking**: prohibirle a la red las jugadas ilegales.
- **Reward shaping**: propina densa además del +1/−1 final, para aprender más
  rápido (con cuidado: puede enseñar trucos malos).
- **γ (gamma)**: 0.995 — las recompensas cercanas valen casi como las
  lejanas, porque la mano es corta.
