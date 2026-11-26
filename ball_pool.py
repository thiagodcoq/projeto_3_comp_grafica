"""Piscina de bolas: simulação física simples feita na CPU (numpy)."""
import numpy as np

GRAVITY = np.array([0.0, -9.8, 0.0])
RESTITUTION = 0.6     # fração da velocidade que sobra depois de uma batida
REST_SPEED = 0.5      # batidas mais lentas que isso não quicam (deixa as bolas pararem)
FRICTION = 2.0        # atrito com o chão (freia o movimento horizontal, por segundo)
SUBSTEP = 1 / 240     # passo máximo de integração (s); passos menores = mais estável
ITERATIONS = 3        # vezes que as colisões são resolvidas em cada subpasso

# cores típicas de piscina de bolas
COLORS = [(0.9, 0.1, 0.08), (0.95, 0.75, 0.05), (0.08, 0.3, 0.9),
          (0.1, 0.7, 0.2), (0.95, 0.4, 0.03), (0.7, 0.15, 0.75)]


class BallPool:
    def __init__(self, n, radius, center, half):
        """n bolas de raio `radius` numa piscina quadrada de lado 2*half centrada em center=(x, z)."""
        self.n, self.r = n, radius
        cx, cz = center
        # limites para o CENTRO das bolas (paredes internas descontando o raio); sem teto
        self.lo = np.array([cx - half + radius, radius, cz - half + radius])
        self.hi = np.array([cx + half - radius, np.inf, cz + half - radius])
        self.colors = [COLORS[i % len(COLORS)] for i in range(n)]
        self.rng = np.random.default_rng(3)
        self.reset()

    def reset(self):
        """Solta todas as bolas de alturas aleatórias acima da piscina."""
        x = self.rng.uniform(self.lo[0], self.hi[0], self.n)
        z = self.rng.uniform(self.lo[2], self.hi[2], self.n)
        y = self.rng.uniform(1.5, 5.0, self.n)
        self.pos = np.column_stack([x, y, z])
        self.vel = np.zeros((self.n, 3))

    def shake(self):
        """Joga as bolas para cima com velocidades aleatórias."""
        self.vel += self.rng.uniform((-1.5, 2.5, -1.5), (1.5, 5.5, 1.5), (self.n, 3))

    def step(self, dt):
        """Avança a simulação dt segundos, dividindo em subpassos pequenos."""
        steps = max(1, int(np.ceil(dt / SUBSTEP)))
        h = dt / steps
        for _ in range(steps):
            # Euler semi-implícito: atualiza a velocidade e depois a posição com a nova velocidade
            self.vel += GRAVITY * h
            self.pos += self.vel * h
            # corrigir um par de bolas pode criar sobreposição com outra (pilhas),
            # então as colisões são resolvidas algumas vezes
            for _ in range(ITERATIONS):
                self.collide_balls()
                self.collide_walls()
            # atrito: bolas encostadas no chão perdem velocidade horizontal
            on_floor = self.pos[:, 1] <= self.r + 1e-3
            self.vel[on_floor] *= (1 - FRICTION * h, 1, 1 - FRICTION * h)

    def collide_walls(self):
        """Chão e paredes: empurra a bola de volta para dentro e rebate a velocidade."""
        below = (self.pos < self.lo) & (self.vel < 0)   # atravessou o chão / parede mínima
        above = (self.pos > self.hi) & (self.vel > 0)   # atravessou a parede máxima
        self.pos = np.clip(self.pos, self.lo, self.hi)
        hit = below | above
        self.vel[hit] *= np.where(np.abs(self.vel[hit]) > REST_SPEED, -RESTITUTION, 0.0)

    def collide_balls(self):
        """Colisão entre pares de bolas (todas as colisões tratadas de uma vez com numpy)."""
        d = self.pos[:, None, :] - self.pos[None, :, :]          # vetor j -> i para cada par
        dist = np.linalg.norm(d, axis=2)
        i, j = np.nonzero(np.triu(dist < 2 * self.r, k=1))       # pares (i < j) que se tocam
        if len(i) == 0:
            return
        dij = np.maximum(dist[i, j], 1e-9)[:, None]
        nrm = d[i, j] / dij                                      # normal do contato
        # 1) separa as bolas: cada uma anda metade da sobreposição
        push = 0.5 * (2 * self.r - dij) * nrm
        np.add.at(self.pos, i, push)
        np.add.at(self.pos, j, -push)
        # 2) impulso (massas iguais): só se estão se aproximando ao longo da normal
        vn = np.sum((self.vel[i] - self.vel[j]) * nrm, axis=1)
        e = np.where(vn < -REST_SPEED, RESTITUTION, 0.0)
        imp = (-0.5 * (1 + e) * np.minimum(vn, 0.0))[:, None] * nrm
        np.add.at(self.vel, i, imp)
        np.add.at(self.vel, j, -imp)
