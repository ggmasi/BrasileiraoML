import pandas as pd
import numpy as np

# --- carrega e limpa stats ---
stats = pd.read_csv("campeonato-brasileiro-estatisticas-full.csv")
stats = stats[stats["posse_de_bola"].notna()].copy()
for col in ["posse_de_bola", "precisao_passes"]:
    if stats[col].dtype == object:
        stats[col] = stats[col].str.rstrip("%").astype(float)

# uma partida é impossível de terminar com 0% de precisao de passes e/ou 0 passes de um time, portanto, se ocorrer um desses casos,
# é uma inconsistencia de dados, assim estes zeros são substituídos por NaN
cols_suspeitas = ["precisao_passes", "passes"]
for col in cols_suspeitas:
    stats[col] = stats[col].replace(0, np.nan)

# --- carrega jogos e renomeia a chave pra bater com stats ---
jogos = pd.read_csv("campeonato-brasileiro-full.csv")
jogos = jogos.rename(columns={"ID": "partida_id"})

# mantém só o que interessa do arquivo de jogos
jogos = jogos[["partida_id", "data", "hora", "mandante", "visitante",
               "mandante_Placar", "visitante_Placar",
               "mandante_Estado", "visitante_Estado", "arena"]].copy()

# junta jogos + stats e fica só com a linha onde o clube é o mandante
df_mandante = jogos.merge(stats, on="partida_id")
df_mandante = df_mandante[df_mandante["clube"] == df_mandante["mandante"]].copy()

# renomeia as colunas de stats com sufixo _mandante
cols_stats = ["chutes", "chutes_no_alvo", "posse_de_bola", "passes",
              "precisao_passes", "faltas", "escanteios"]
df_mandante = df_mandante.rename(columns={c: c + "_mandante" for c in cols_stats})
df_mandante = df_mandante.drop(columns=["clube", "rodata"])

df_visitante = jogos[["partida_id", "visitante"]].merge(stats, on="partida_id")
df_visitante = df_visitante[df_visitante["clube"] == df_visitante["visitante"]].copy()
df_visitante = df_visitante.rename(columns={c: c + "_visitante" for c in cols_stats})
df_visitante = df_visitante[["partida_id"] + [c + "_visitante" for c in cols_stats]]

# une os dois lados
df = df_mandante.merge(df_visitante, on="partida_id")

df["saldo"] = df["mandante_Placar"] - df["visitante_Placar"]
df["resultado"] = np.select(
    [df["saldo"] > 0, df["saldo"] < 0],
    ["C", "V"],
    default="E"
)

for lado in ["mandante", "visitante"]:
    for col in ["posse_de_bola", "precisao_passes"]:
        nome = f"{col}_{lado}"
        df[nome] = (
            df[nome].astype(str)          # garante texto
                    .str.rstrip("%")       # tira o %
                    .replace("nan", np.nan)
                    .astype(float)          # vira número
        )

cols_stats = ["chutes", "chutes_no_alvo", "posse_de_bola", "passes",
              "precisao_passes", "faltas", "escanteios"]

# --- visão do MANDANTE ---
log_mandante = pd.DataFrame({
    "partida_id": df["partida_id"],
    "data": df["data"],
    "time": df["mandante"],
    "adversario": df["visitante"],
    "mando": "casa",                          # esse time jogou em casa
    "arena": df["arena"],
    "estado_time": df["mandante_Estado"],
    "estado_adversario": df["visitante_Estado"],
    "gols_feitos": df["mandante_Placar"],
    "gols_sofridos": df["visitante_Placar"],
})
# estatísticas do próprio time (as do mandante) e as sofridas (as do visitante)
for c in cols_stats:
    log_mandante[c] = df[f"{c}_mandante"]
    log_mandante[f"{c}_sofrido"] = df[f"{c}_visitante"]

# --- visão do VISITANTE ---
log_visitante = pd.DataFrame({
    "partida_id": df["partida_id"],
    "data": df["data"],
    "time": df["visitante"],
    "adversario": df["mandante"],
    "mando": "fora",                          # esse time jogou fora
    "arena": df["arena"],
    "estado_time": df["visitante_Estado"],
    "estado_adversario": df["mandante_Estado"],
    "gols_feitos": df["visitante_Placar"],
    "gols_sofridos": df["mandante_Placar"],
})
for c in cols_stats:
    log_visitante[c] = df[f"{c}_visitante"]
    log_visitante[f"{c}_sofrido"] = df[f"{c}_mandante"]

# empilha: agora cada partida vira 2 linhas (uma por time)
log = pd.concat([log_mandante, log_visitante], ignore_index=True)

# deriva o resultado da ÓTICA daquele time (vitória/empate/derrota dele)
log["saldo"] = log["gols_feitos"] - log["gols_sofridos"]
log["resultado_time"] = np.select(
    [log["saldo"] > 0, log["saldo"] < 0],
    ["V", "D"],          # V = venceu, D = derrota (deste time)
    default="E"
)

log["data"] = pd.to_datetime(log["data"], format="%d/%m/%Y")
log = log.sort_values(["time", "data"]).reset_index(drop=True)

# em partidas de 2014 até o fim de 2015 os chutes ao alvos são todos 0, o que é uma inconsistência, assim, os valores dessa
# coluna neste período são considerados NaN para não afetar o resultado dos modelos
ID_INICIAL = 4740
ID_FINAL = 5742
ids_falhos = log["partida_id"].between(ID_INICIAL, ID_FINAL)
for col in ["chutes_no_alvo"]:
    log[col] = np.where(ids_falhos & (log[col] == 0), np.nan, log[col])


log["media_chutes_5"] = (
    log.groupby("time")["chutes"]
       .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
)

fla = log[log["time"] == "Flamengo"][["data", "chutes", "media_chutes_5"]].head(7)

bases = ["chutes", "chutes_no_alvo", "posse_de_bola", "passes",
         "precisao_passes", "faltas", "escanteios",
         "gols_feitos", "gols_sofridos"]

JANELA = 5

def media_movel(serie):
    return serie.shift(1).rolling(JANELA, min_periods=1).mean()

for col in bases:
    log[f"{col}_media5"] = log.groupby("time")[col].transform(media_movel)

log["saldo_media5"] = log["gols_feitos_media5"] - log["gols_sofridos_media5"]

log["pontos"] = log["resultado_time"].map({"V": 3, "E": 1, "D": 0})
log["forma_pontos5"] = log.groupby("time")["pontos"].transform(
    lambda s: s.shift(1).rolling(5, min_periods=1).sum()
)

# nenhuma feature media5 pode ter vazado: a 1ª aparição de cada time deve ser NaN
primeiras = log.groupby("time").head(1)

def media_movel(serie):
    return serie.shift(1).rolling(5, min_periods=1).mean()

# desempenho condicionado ao mando (casa usa só jogos em casa; fora, só fora)
for col in ["chutes", "gols_feitos", "gols_sofridos", "escanteios"]:
    log[f"{col}_media5_mando"] = (
        log.groupby(["time", "mando"])[col].transform(media_movel)
    )

log["saldo_media5_mando"] = log["gols_feitos_media5_mando"] - log["gols_sofridos_media5_mando"]

log["pontos_na_arena"] = (
    log.groupby(["time", "arena"])["pontos"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=1).mean()
    )
)

log["saldo_h2h"] = (
    log.groupby(["time", "adversario"])["saldo"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )
)
log["pontos_h2h"] = (
    log.groupby(["time", "adversario"])["pontos"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )
)

log["classico_estadual"] = (log["estado_time"] == log["estado_adversario"]).astype(int)

# features de histórico que viram entrada do modelo (todas pré-jogo, livres de vazamento)
features = [
    "chutes_media5", "chutes_no_alvo_media5", "posse_de_bola_media5",
    "passes_media5", "precisao_passes_media5", "faltas_media5", "escanteios_media5",
    "gols_feitos_media5", "gols_sofridos_media5", "saldo_media5",
    "forma_pontos5",
    "chutes_media5_mando", "gols_feitos_media5_mando", "gols_sofridos_media5_mando",
    "escanteios_media5_mando", "saldo_media5_mando",
    "pontos_na_arena", "saldo_h2h", "pontos_h2h",
]

# lado do mandante (jogou em casa)
casa = log[log["mando"] == "casa"].copy()
casa = casa[["partida_id", "data", "time", "adversario", "classico_estadual"] + features]
casa = casa.rename(columns={"time": "mandante", "adversario": "visitante"})
casa = casa.rename(columns={f: f"{f}_mand" for f in features})

# lado do visitante (jogou fora) — só features, o resto já vem do lado casa
fora = log[log["mando"] == "fora"].copy()
fora = fora[["partida_id"] + features]
fora = fora.rename(columns={f: f"{f}_vis" for f in features})

# junta os dois lados: uma linha por jogo
base = casa.merge(fora, on="partida_id")

# recupera o alvo da partida (C/V/E) que montamos no passo 2
base = base.merge(df[["partida_id", "resultado"]], on="partida_id")

# diferenças mandante - visitante: é aqui que mora o "quem finaliza mais ganha"
for f in features:
    base[f"dif_{f}"] = base[f"{f}_mand"] - base[f"{f}_vis"]

# IMPUTAÇÃO NEUTRA: inclui dados "neutros" para casos inicias (como as primeiras rodadas de uma temporada, ou primeira encontro de duas equipes)
cols_zero = [col for col in base.columns if "saldo" in col or col.startswith("dif_")]
for col in cols_zero:
    base[col] = base[col].fillna(0)

cols_pontos = [col for col in base.columns if "pontos" in col]
for col in cols_pontos:
    base[col] = base[col].fillna(1.0)

base.to_csv("base_modelo.csv", index=False)