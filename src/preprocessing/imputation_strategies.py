from abc import ABC, abstractmethod


class ImputationStrategy(ABC):
    """
    Interfaccia comune del pattern Strategy.

    Ogni strategia concreta deve implementare lo stesso metodo `imputa`,
    cosi il main puo scegliere l'algoritmo a runtime senza conoscere i
    dettagli interni dell'imputazione.
    """

    codice_menu = ""
    nome_menu = ""
    nome_report = ""

    @abstractmethod
    def imputa(self, missing_handler, train_values, test_values, colonna):
        """Imputa i valori mancanti della colonna indicata su train e test."""
        pass


class MeanImputationStrategy(ImputationStrategy):
    """Strategia concreta: sostituisce i NaN con la media calcolata sul train."""

    codice_menu = "1"
    nome_menu = "Univariata - Media"
    nome_report = "univariata_media"

    def imputa(self, missing_handler, train_values, test_values, colonna):
        return missing_handler.imputa_univariata_media(
            train_df=train_values,
            test_df=test_values,
            colonna=colonna,
        )


class MedianImputationStrategy(ImputationStrategy):
    """Strategia concreta: sostituisce i NaN con la mediana calcolata sul train."""

    codice_menu = "2"
    nome_menu = "Univariata - Mediana"
    nome_report = "univariata_mediana"

    def imputa(self, missing_handler, train_values, test_values, colonna):
        return missing_handler.imputa_univariata_mediana(
            train_df=train_values,
            test_df=test_values,
            colonna=colonna,
        )


class LinearRegressionImputationStrategy(ImputationStrategy):
    """Strategia concreta: stima i NaN con una regressione lineare sui predittori numerici."""

    codice_menu = "3"
    nome_menu = "Multivariata - Regressione lineare"
    nome_report = "multivariata_regressione_lineare"

    def imputa(self, missing_handler, train_values, test_values, colonna):
        return missing_handler.imputa_multivariata_regressione_lineare(
            train_df=train_values,
            test_df=test_values,
            colonna=colonna,
        )


class KnnImputationStrategy(ImputationStrategy):
    """Strategia concreta: stima i NaN usando un KNN Regressor sui predittori numerici."""

    codice_menu = "4"
    nome_menu = "KNN predictor"
    nome_report = "knn_predictor"

    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def imputa(self, missing_handler, train_values, test_values, colonna):
        return missing_handler.imputa_knn_predictor(
            train_df=train_values,
            test_df=test_values,
            colonna=colonna,
            n_neighbors=self.n_neighbors,
        )


class ImputationContext:
    """
    Context del pattern Strategy.

    Mantiene la strategia corrente e delega a essa l'imputazione. In questo
    modo il codice chiamante lavora sempre con lo stesso metodo
    `imputa_colonna`, indipendentemente dall'algoritmo selezionato.
    """

    def __init__(self, strategy):
        self.strategy = strategy

    def set_strategy(self, strategy):
        self.strategy = strategy

    def imputa_colonna(self, missing_handler, train_values, test_values, colonna):
        return self.strategy.imputa(
            missing_handler=missing_handler,
            train_values=train_values,
            test_values=test_values,
            colonna=colonna,
        )


STRATEGIE_IMPUTAZIONE = {
    # Registry delle strategie disponibili: il codice del menu diventa la chiave
    # per recuperare l'oggetto Strategy da usare nel Context.
    strategy.codice_menu: strategy
    for strategy in (
        MeanImputationStrategy(),
        MedianImputationStrategy(),
        LinearRegressionImputationStrategy(),
        KnnImputationStrategy(n_neighbors=5),
    )
}

CODICE_STRATEGIA_DA_NOME_REPORT = {
    # Mappa inversa usata quando la scelta automatica KNN restituisce il nome
    # della strategia migliore invece del codice digitato dall'utente.
    strategy.nome_report: codice
    for codice, strategy in STRATEGIE_IMPUTAZIONE.items()
}


def applica_strategia_imputazione_colonna(missing_handler, train_values, test_values, scelta, colonna):
    """
    Punto di ingresso usato dal main.

    Riceve la scelta dell'utente o della valutazione automatica, seleziona la
    Strategy corrispondente e la esegue tramite il Context.
    """
    if scelta not in STRATEGIE_IMPUTAZIONE:
        raise ValueError(f"Scelta strategia non valida: {scelta}")

    context = ImputationContext(STRATEGIE_IMPUTAZIONE[scelta])
    return context.imputa_colonna(
        missing_handler=missing_handler,
        train_values=train_values,
        test_values=test_values,
        colonna=colonna,
    )
