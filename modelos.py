import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# PREPARAÇÂO UNIVERSAL

# carrega os dados gerados
df = pd.read_csv("base_modelo.csv")

# apenas as "diferenças" e se é um clássico são interessantes para o modelo
features = [col for col in df.columns if col.startswith("dif_")] + ["classico_estadual"]
X = df[df.columns.intersection(features)] # filtro que garante que a coluna existe
y = df["resultado"]

# separa o que é treino (passado) e o que é teste (futuro)
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, shuffle=False)


# dicionário com o nome de cada modelo e o algoritmo que este modelo utiliza
modelos = {
    "Regressao_Logistica": LogisticRegression(max_iter=1000, class_weight='balanced'),

    "Arvore_de_Decisao": DecisionTreeClassifier(class_weight='balanced', random_state=42),

    "Floresta_Aleatoria": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),

    "k-NN": KNeighborsClassifier(n_neighbors=5)
}

# um arquivo com o boletim de cada algoritmo é gerado
with open("relatorio.txt", "w", encoding="utf-8") as arquivo_relatorio:
    
    for nome_modelo, algoritmo in modelos.items():

        # criação de um pipeline para padronização dos dados
        pipeline = Pipeline([
            ('tapa_buracos', SimpleImputer(strategy='median')), # transforma NaNs em valores neutros
            ('padronizador', StandardScaler()), # coloca as métricas num mesmo padrão
            ('modelo', algoritmo) # aplicação do algoritmo da vez
        ])

        # treina e prevê
        pipeline.fit(X_treino, y_treino)
        previsoes = pipeline.predict(X_teste)

        # salva o boletim no relatório
        arquivo_relatorio.write(f"==================================================\n")
        arquivo_relatorio.write(f"MODELO: {nome_modelo}\n\n")
        arquivo_relatorio.write(classification_report(y_teste, previsoes) + "\n\n")



        # método para mostrar "ranking" de importância de cada coluna
        modelo_treinado = pipeline.named_steps['modelo']
        pesos = None

        # checa se é regressaão logística
        if hasattr(modelo_treinado, 'coef_'):
            pesos = np.abs(modelo_treinado.coef_).mean(axis=0)
        # checa se é um modelo de árvore
        elif hasattr(modelo_treinado, 'feature_importances_'):
            pesos = modelo_treinado.feature_importances_

        if pesos is not None:
            ranking = pd.DataFrame({
                'Estatística': features,
                'Peso': pesos
            }).sort_values(by='Peso', ascending=False).reset_index(drop=True)

            arquivo_relatorio.write("=== RANKING DE CRITÉRIOS ===\n")
            arquivo_relatorio.write(ranking.to_string() + "\n\n")
        else:
            arquivo_relatorio.write("=== RANKING DE CRITÉRIOS ===\n")
            arquivo_relatorio.write("Este algoritmo não fornece pesos das estatísticas.\n\n")

        arquivo_relatorio.write(f"==================================================\n")
        # gera a planilha de auditoria do modelo atual
        indices_teste = X_teste.index
        df_auditoria = df.loc[indices_teste, ['data', 'mandante', 'visitante']].copy()
        df_auditoria['Resultado_Real'] = y_teste.values
        df_auditoria['Previsoes_Modelo'] = previsoes

        # se o modelo permitir, extrai as possibilidades
        if hasattr(pipeline, "predict_probability"):
            probabilidades = pipeline.predict_proba(X_teste)
            for i, classe in enumerate(pipeline.classes_):
                df_auditoria[f'Prob_{classe}'] = probabilidades[:, i]

        df_auditoria['Acertou?'] = (df_auditoria['Resultado_Real'] == df_auditoria['Previsoes_Modelo']).map({True: "SIM", False: "NÃO"})
 
        # salva o CSV com o nome do modelo
        df_auditoria.to_csv(f"auditoria_{nome_modelo}.csv", index=False)

print("\nTestes finalizados!\n")
