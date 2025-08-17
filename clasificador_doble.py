import sys
import os
import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import ttest_ind

def convert_bytes_to_str(df):
    """Convierte columnas de bytes a strings"""
    for col in df.columns:
        if df[col].dtype == object:
            # Manejar diferentes tipos de datos categóricos
            if isinstance(df[col].iloc[0], bytes):
                df[col] = df[col].str.decode('utf-8')
            elif isinstance(df[col].iloc[0], str):
                # Convertir 'b'0'' a '0' si es necesario
                df[col] = df[col].str.replace("b'", "").str.replace("'", "")
    return df

def main():
    if len(sys.argv) < 2:
        print("Uso: python3 clasificador_manual.py <archivo_manual.arff>")
        print("Ejemplo: python3 clasificador_manual.py driving_data_manual.arff")
        sys.exit(1)
        
    training_file = sys.argv[1]

    # Cargar datos
    print(f"\nCargando datos: {training_file}")
    try:
        data_train, meta_train = arff.loadarff(training_file)
        df_train = pd.DataFrame(data_train)
        df_train = convert_bytes_to_str(df_train)
    except Exception as e:
        print(f"Error cargando archivo: {str(e)}")
        sys.exit(1)
    
    # Verificar estructura de datos manuales
    required_sensor_columns = ['s_left', 's_lc', 's_c', 's_rc', 's_right']
    required_action_columns = ['left', 'right']
    
    # Verificar sensores
    sensor_cols = [col for col in required_sensor_columns if col in df_train.columns]
    if len(sensor_cols) < 3:
        print("\nERROR: El archivo no tiene suficientes columnas de sensores")
        print("Columnas encontradas:", df_train.columns.tolist())
        print("Se esperan al menos 3 de:", required_sensor_columns)
        sys.exit(1)
    
    # Verificar acciones
    action_cols = [col for col in required_action_columns if col in df_train.columns]
    if len(action_cols) < 2:
        print("\nERROR: El archivo no tiene columnas de acciones")
        print("Columnas encontradas:", df_train.columns.tolist())
        print("Se esperan:", required_action_columns)
        sys.exit(1)
    
    # Preprocesamiento específico para datos manuales
    # Convertir acciones a categorías combinadas
    try:
        # Convertir a numérico si es necesario
        for col in action_cols:
            df_train[col] = pd.to_numeric(df_train[col], errors='coerce').fillna(0).astype(int)
        
        conditions = [
            (df_train['left'] == 1) & (df_train['right'] == 0),
            (df_train['left'] == 0) & (df_train['right'] == 1),
            (df_train['left'] == 0) & (df_train['right'] == 0)
        ]
        choices = ['left', 'right', 'straight']
        df_train['action'] = np.select(conditions, choices, default='unknown')
        
        # Eliminar filas con acciones desconocidas
        df_train = df_train[df_train['action'] != 'unknown']
        
        if len(df_train) == 0:
            print("\nERROR: No hay datos válidos después del filtrado")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error procesando acciones: {str(e)}")
        sys.exit(1)
    
    # Separar características y objetivo
    X = df_train[sensor_cols].astype(float)
    y = df_train['action']
    
    # Modelos de clasificación
    models = {
        "Decision_Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random_Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        "KNN_k=3": KNeighborsClassifier(n_neighbors=3),
        "KNN_k=5": KNeighborsClassifier(n_neighbors=5),
        "Naive_Bayes": GaussianNB(),
        "ZeroR": DummyClassifier(strategy='most_frequent')
    }
    
    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("\nRESULTADOS DE VALIDACIÓN CRUZADA (5 folds):")
    results = {}
    for name, model in models.items():
        try:
            scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            mean_acc = np.mean(scores)
            std_acc = np.std(scores)
            results[name] = scores
            print(f"{name}: {mean_acc:.4f} ± {std_acc:.4f}")
        except Exception as e:
            print(f"Error evaluando {name}: {str(e)}")
    
    # Test estadístico
    base_model = "ZeroR"
    print("\nTEST ESTADÍSTICO (t-test):")
    if base_model in results:
        for model_name in models.keys():
            if model_name == base_model or model_name not in results:
                continue
            try:
                t_stat, p_value = ttest_ind(results[base_model], results[model_name])
                print(f"{base_model} vs {model_name}: t = {t_stat:.4f}, p = {p_value:.4f}")
            except Exception as e:
                print(f"Error en t-test: {str(e)}")
    
    # Evaluación con conjunto de prueba
    print("\nEVALUACIÓN EN CONJUNTO DE PRUEBA:")
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            print(f"\n{name}:")
            print(f"Exactitud: {acc:.4f}")
            print("Reporte de clasificación:")
            print(classification_report(y_test, y_pred))
        except Exception as e:
            print(f"Error evaluando {name}: {str(e)}")
        
    # Análisis de características
    print("\nANÁLISIS DE CARACTERÍSTICAS:")
    print("Medias de sensores por acción:")
    df_analysis = df_train.copy()
    df_analysis[sensor_cols] = df_analysis[sensor_cols].astype(float)
    print(df_analysis.groupby('action')[sensor_cols].mean())
    
    # Verificar posibles problemas de datos
    print("\nVERIFICACIÓN DE DATOS:")
    print(f"Total de muestras: {len(df_train)}")
    print("Distribución de acciones:")
    print(y.value_counts(normalize=True))
    
    # Verificar correlaciones
    corr_matrix = df_train[sensor_cols].corr()
    print("\nMatriz de correlación entre sensores:")
    print(corr_matrix)
    
    # Verificar valores faltantes
    print("\nVALORES FALTANTES:")
    print(df_train.isnull().sum())

if __name__ == "__main__":
    main()