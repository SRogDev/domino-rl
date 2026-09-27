# Domino RL 🁫

Enseñarle a una IA a jugar **dominó cubano de 9** (doble-9, 55 fichas, parejas)
usando **Reinforcement Learning** — y de paso, aprender RL desde cero.

> Estado: Fases 1–2 listas (simulador, entorno Gymnasium, baselines, evaluación
> y script de entrenamiento). El entrenamiento real (Fase 5) es el siguiente paso.

## La idea en 30 segundos

En *supervised learning* entrenas con ejemplos etiquetados. En *reinforcement
learning* no hay etiquetas: hay un **agente** que juega, un **entorno** (este
simulador) que le dice el **estado** del juego, y una **recompensa** (+1 ganar,
−1 perder) al final de cada mano. El agente juega millones de manos contra sí
mismo (*self-play*) y ajusta su **política** (qué jugada elegir en cada estado)
para maximizar la recompensa. Sin GPU: la red es diminuta y el simulador es
Python puro — la capa gratuita de Colab sobra.

## Estructura

```
src/domino_rl/
  game.py       Motor del dominó cubano doble-9 (reglas, reparto, tantos)
  env.py        Entorno estilo Gymnasium: obs[143] (incluye historial público
                de fichas jugadas — el conteo debe emerger, no hardcodearse),
                111 acciones con máscara, reward sparse por defecto (bitter lesson)
  baselines.py  Agentes de referencia: random y greedy (heurístico)
  evaluate.py   Torneos entre políticas: win-rate, puntos, dominó vs tranca
  train.py      Entrenamiento PPO con self-play (CPU, sin GPU)
  demo/         Mesa digital: humanos (móvil) vs IA —
                `PYTHONPATH=src python demo/server.py`, abrir
                http://<ip>:8000 en cada teléfono
tests/          Tests del motor (TDD: se escribieron antes que el motor)
PLAN.md         El plan paso a paso completo (Fases 0–8)
```

## Uso rápido

```bash
uv sync --group train && source .venv/bin/activate

# 1. Ver que el simulador funciona: torneo greedy vs random
PYTHONPATH=src python -m domino_rl.evaluate --rounds 200 --a greedy --b random

# 2. Entrenar parejas 2v2 con IPPO (CPU; en Colab gratis funciona igual)
#     (--iters = iteraciones TOTALES; LR y entropía decaen linealmente)
PYTHONPATH=src python -m domino_rl.ippo --iters 1000 --manos-per-iter 256 \
  --mode teams --n-players 4 --match-target 100 \
  --lr 1e-4 --lr-end 0.0 --ent-coef 0.1 --ent-end 0.01 \
  --epochs 2 --seed 42 \
  --out checkpoints/ippo_2v2 --ckpt-every 100 --eval-every 100 --eval-manos 200

# 3. Medir al campeón contra el heurístico (500 manos, ±2%)
PYTHONPATH=src python -m domino_rl.evaluate --rounds 500 \
  --a ippo:checkpoints/mappo_2v2/ckpt_500.pt --b greedy \
  --mode teams --n-players 4

# 4. Ver CÓMO juega, no solo cuánto gana (stats de estilo + transcripciones)
PYTHONPATH=src python -m domino_rl.analyze \
  --policy checkpoints/mappo_2v2/ckpt_500.pt \
  --manos 200 --opponent greedy --transcripts 3
```

## Continuar un entrenamiento en otra máquina

Todo lo necesario está en el repo: código, `pyproject.toml` + `uv.lock`
(dependencias) y el checkpoint `checkpoints/mappo_2v2/ckpt_500.pt`
(53.8% vs pareja greedy, 59.8% vs random). Solo CPU — la red tiene ~35k
parámetros, no hace falta GPU.

```bash
git clone https://github.com/SRogDev/domino-rl.git && cd domino-rl
uv sync --group train && source .venv/bin/activate

# retomar desde el checkpoint 500: restaura pesos + optimizador + RNG + iter
# (los checkpoints viejos de solo-pesos retoman "tibio": pesos sí, resto no).
# --iters es el TOTAL: 520 = 20 iteraciones más desde el 500.
PYTHONPATH=src python -m domino_rl.ippo --iters 520 --manos-per-iter 256 \
  --mode teams --n-players 4 --match-target 100 \
  --lr 1e-4 --lr-end 0.0 --ent-coef 0.1 --ent-end 0.01 \
  --epochs 2 --seed 42 \
  --resume checkpoints/mappo_2v2/ckpt_500.pt \
  --out checkpoints/ippo_2v2_cont --ckpt-every 20 --eval-every 20 --eval-manos 200
```

## Reglas implementadas

Doble-9 (55 fichas), 4 jugadores × 10 fichas (15 quedan en el pozo), sale el
doble más alto en mano (obligado), por parejas (0&2 vs 1&3) o todos contra
todos (`--mode individual`), gana la mano por dominó o por tranca, tantos =
pips restantes del contrario. Detalles y decisiones de diseño en `PLAN.md`.

## Roadmap

Ver **[PLAN.md](PLAN.md)**: 9 fases desde "no sé nada de RL" hasta una IA que
te gana al dominó, con qué aprender en cada fase y cómo saber que vas bien.
