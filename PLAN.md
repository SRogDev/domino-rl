# Plan: de cero a una IA campeona de dominó cubano 🁫

Meta: aprender Reinforcement Learning construyendo un agente que juegue
**dominó cubano de 9** (doble-9, 55 fichas, parejas) a gran nivel.
Restricción: **sin GPU potente** — solo CPU y las ~30h gratis de Colab.

Buenas noticias primero: **no necesitas GPU para esto.** La red que vamos a
entrenar es diminuta (98 → 128 → 128 → 111 neuronas) y el simulador es Python
puro y rapidísimo. El cuello de botella en RL de juegos no es el hardware,
es el diseño: cómo representas el estado, cómo defines la recompensa y contra
quién entrena el agente. Todo eso corre perfecto en CPU.

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

- **Observación** (`env.py::observe`): vector de 98 números — mi mano (55),
  puntas de la mesa (20), pips ya jugados (10), fichas restantes por jugador
  (4), pases seguidos (1), quién soy y quién es mi pareja (8). Todo
  normalizado a [0,1]. Pregunta clave de RL: ¿qué información necesita el
  agente para decidir bien? (Ej: saber qué pips ya salieron permite "contar".)
- **Acciones** (111): 55 fichas × 2 puntas + pasar. Con **máscara de acciones
  legales**: la red nunca elige una jugada ilegal (truco estándar que acelera
  muchísimo el aprendizaje).
- **Recompensa**: `sparse` (+1/−1/0 al final) o `shaped` (más un empujoncito
  denso por dejar menos pips que el contrario). Empieza con `shaped`.

**Qué aprender aquí:** el 80% del éxito en RL aplicado está en estos tres
diseños, no en el algoritmo. Si el agente no aprende, casi siempre el problema
está aquí.

**Listo cuando:** puedas explicar por qué cada bloque de 98 números está ahí.

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

**Objetivo:** PPO con red neuronal jugando contra sí mismo en doble-9.

Ya tienes el script: `src/domino_rl/train.py` (usa `MaskablePPO` de
sb3-contrib). El truco clave ya está implementado: **self-play con política
compartida** — el entorno sienta al mismo agente en las 4 sillas y cada
recompensa se calcula desde la perspectiva del equipo que actúa. Así un PPO
monojugador entrena un juego de 4.

```bash
# En Colab (gratis, CPU): clona el repo, pip install -e ".[train]" y:
python -m domino_rl.train --timesteps 2000000 --n-envs 4 --out /content/drive/MyDrive/domino-rl
```

- 2M de pasos ≈ unas horas en CPU de Colab. Guarda checkpoints.
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

## Fase 7 — Subir de nivel (iterativo, el resto del proyecto)

Ideas ordenadas por impacto esperado. Ataca la que revele la Fase 6:

1. **Mejor recompensa/observación**: añade "pips que le quedan al contrario
   por valor" o historial de jugadas de la pareja (memoria corta).
2. **League play**: entrena contra un *pool* de versiones pasadas del agente
   (no solo contra sí mismo actual) → evita estrategias cíclicas.
3. **Parejas de verdad**: hoy el agente no sabe cooperar explícitamente;
   recompensa de equipo + observación de la pareja ya lo incentivan, pero se
   puede modelar mejor (p. ej. entrenar con comunicación implícita vía jugadas
   "señal").
4. **MCTS en inferencia**: en tu turno, simula N futuros con el modelo actual
   y elige la mejor jugada (estilo AlphaZero-lite, solo CPU).
5. **Currículo**: entrena primero en `mode="individual"` (más simple, sin
   problema de asignación de crédito en equipo) y luego transfiere a parejas.

## Fase 8 — Jugar contra ella (1 semana, la diversión)

**Objetivo:** una web mínima donde retes a tu agente.

Next.js + un endpoint Python que carga `final.zip` y responde jugadas.
Nada de GPU en servidor: la red es tan pequeña que infiere en milisegundos en
CPU. (Si quieres, esto puede ser tu proyecto #N del laboratorio.)

---

## Decisiones de diseño ya tomadas (y por qué)

- **Episodio = una mano**, no un partido a 100: episodios cortos = más
  señal de aprendizaje por hora. Los tantos del partido se añaden después.
- **Primero `teams` directo**: el dominó cubano es de parejas; el self-play
  con política compartida lo maneja sin complejidad extra.
- **Sin GPU por diseño**: si algún día quieres escalar, el mismo código corre
  en GPU sin cambios (PyTorch lo hace solo).
- **Reglas simplificadas a propósito**: sin capicúa/bonificaciones raras.
  Se añaden cuando el agente domine lo básico.

## Glosario rápido

- **Episodio**: una mano completa, del reparto al conteo de tantos.
- **Política (π)**: la red neuronal; entra el estado (98 números), sale la
  jugada.
- **Self-play**: el agente es su propio rival en las 4 sillas.
- **Action masking**: prohibirle a la red las jugadas ilegales.
- **Reward shaping**: propina densa además del +1/−1 final, para aprender más
  rápido (con cuidado: puede enseñar trucos malos).
- **γ (gamma)**: 0.995 — las recompensas cercanas valen casi como las
  lejanas, porque la mano es corta.
