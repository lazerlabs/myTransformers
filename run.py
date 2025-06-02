import click


@click.command()

# Basic Config
@click.option("--is_training", type=int, required=True, default=1, help="Training status")
@click.option("--model_id", type=str, required=True, default="test", help="Model ID")

# Data Loader
@click.option('--data-dir', type=str, help='Directory containing data files')
@click.option('--stocks', type=str, default=None, help='Comma-separated list of stock tickers (e.g. AAPL,MSFT)')
@click.option('--features', type=str, default='volume,close' help='Comma-separated list of features (e.g. volume,close,transactions)')
@click.option('--train-size', type=int, help='Number of files to use for training')
@click.option('--test-size', type=int, help='Number of files to use for testing')
@click.option('--val-size', type=int, help='Number of files to use for validation')
@click.option('--val-stocks', type=str, default=None, help='Comma-separated list of validation stock tickers')
