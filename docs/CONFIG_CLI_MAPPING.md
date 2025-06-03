# Configuration and CLI Mapping

This document shows the complete mapping between `configs.py` fields and `train.py` CLI options. **configs.py is the single source of truth for all default values.**

## Architecture

- **Single Source of Truth**: All default values are defined only in `configs.py`
- **CLI Integration**: CLI options automatically inherit defaults from `configs.py` using `get_config_defaults()`
- **Visible Defaults**: CLI help shows the actual default values with `show_default=True`
- **Override Capability**: CLI arguments override config defaults when explicitly provided

## How It Works

1. `get_config_defaults()` extracts default values from the dataclass fields without initializing problematic components
2. CLI decorators use `default=_config_defaults['field_name']` to inherit config defaults
3. `show_default=True` makes these defaults visible in CLI help
4. The main function applies CLI overrides to the config instance

## Benefits

- ✅ **Single Source of Truth**: Change a default in one place (`configs.py`) and it automatically updates everywhere
- ✅ **Visible Defaults**: Users can see actual default values in CLI help
- ✅ **Type Safety**: Dataclass validation and type hints ensure consistency
- ✅ **DRY Principle**: No duplicate default definitions
- ✅ **Maintainable**: Adding new config fields automatically creates corresponding CLI options

## Data Parameters
| Config Field | CLI Option | Default Source | Type |
|-------------|------------|----------------|------|
| `data_dir` | `--data-dir` | `configs.data_dir` | `str` |
| `stocks` | `--stocks` | `None` (special case) | `Optional[List[str]]` |
| `features` | `--features` | `configs.features` | `List[str]` |
| `train_size` | `--train-size` | `configs.train_size` | `Optional[int]` |
| `test_size` | `--test-size` | `configs.test_size` | `int` |
| `val_size` | `--val-size` | `configs.val_size` | `int` |
| `val_stocks` | `--val-stocks` | `configs.val_stocks` | `List[str]` |

## Sequence Parameters
| Config Field | CLI Option | Default Source | Type |
|-------------|------------|----------------|------|
| `seq_len` | `--seq-len` | `configs.seq_len` | `int` |
| `pred_len` | `--pred-len` | `configs.pred_len` | `int` |
| `label_len` | `--label-len` | `configs.label_len` | `int` |
| `scale` | `--scale/--no-scale` | `configs.scale` | `bool` |

## Model Parameters
| Config Field | CLI Option | Default Source | Type |
|-------------|------------|----------------|------|
| `model` | `--model` | `configs.model` | `str` |
| `d_model` | `--d-model` | `configs.d_model` | `int` |
| `n_heads` | `--n-heads` | `configs.n_heads` | `int` |
| `e_layers` | `--e-layers` | `configs.e_layers` | `int` |
| `d_ff` | `--d-ff` | `configs.d_ff` | `int` |
| `dropout` | `--dropout` | `configs.dropout` | `float` |
| `embed` | `--embed` | `configs.embed` | `str` |
| `activation` | `--activation` | `configs.activation` | `str` |
| `output_attention` | `--output-attention/--no-output-attention` | `configs.output_attention` | `bool` |
| `use_norm` | `--use-norm/--no-use-norm` | `configs.use_norm` | `bool` |

## Inference Parameters
| Config Field | CLI Option | Default Source | Type |
|-------------|------------|----------------|------|
| `temperature` | `--temperature` | `configs.temperature` | `float` |

## Training Parameters
| Config Field | CLI Option | Default Source | Type |
|-------------|------------|----------------|------|
| `batch_size` | `--batch-size` | `configs.batch_size` | `int` |
| `learning_rate` | `--learning-rate` | `configs.learning_rate` | `float` |
| `train_epochs` | `--train-epochs` | `configs.train_epochs` | `int` |
| `patience` | `--patience` | `configs.patience` | `int` |
| `max_train_iterations` | `--max-train-iterations` | `configs.max_train_iterations` | `Optional[int]` |

## Checkpoint Parameters
| Config Field | CLI Option | Default Source | Type |
|-------------|------------|----------------|------|
| `save_checkpoint_every_n_iterations` | `--save-checkpoint-every-n-iterations` | `configs.save_checkpoint_every_n_iterations` | `Optional[int]` |
| `save_checkpoint_every_n_epochs` | `--save-checkpoint-every-n-epochs` | `configs.save_checkpoint_every_n_epochs` | `Optional[int]` |

## Loss Parameters
| Config Field | CLI Option | Default Source | Type |
|-------------|------------|----------------|------|
| `loss_type` | `--loss-type` | `configs.loss_type` | `str` |
| `loss_kwargs` | `--loss-kwargs` | `configs.loss_kwargs` (JSON) | `dict` |

## Learning Rate Scheduler Parameters
| Config Field | CLI Option | Default Source | Type |
|-------------|------------|----------------|------|
| `lr_scheduler` | `--lr-scheduler` | `configs.lr_scheduler` | `str` |
| `lr_decay_factor` | `--lr-decay-factor` | `configs.lr_decay_factor` | `float` |
| `lr_patience` | `--lr-patience` | `configs.lr_patience` | `int` |
| `min_lr` | `--min-lr` | `configs.min_lr` | `float` |
| `warmup_epochs` | `--warmup-epochs` | `configs.warmup_epochs` | `int` |

## Device Parameters
| Config Field | CLI Option | Default Source | Type |
|-------------|------------|----------------|------|
| `use_gpu` | `--use-gpu/--no-use-gpu` | `configs.use_gpu` | `bool` |
| `use_multi_gpu` | `--use-multi-gpu/--no-use-multi-gpu` | `configs.use_multi_gpu` | `bool` |
| `gpu` | `--gpu` | `configs.gpu` | `int` |
| `device_ids` | `--device-ids` | `None` (special case) | `Optional[List[int]]` |

## Directory Parameters
| Config Field | CLI Option | Default Source | Type |
|-------------|------------|----------------|------|
| `checkpoints_dir` | `--checkpoints-dir` | `configs.checkpoints_dir` | `str` |
| `logs_dir` | `--logs-dir` | `configs.logs_dir` | `str` |
| `figures_dir` | `--figures-dir` | `configs.figures_dir` | `str` |
| `embeddings_dir` | `--embeddings-dir` | `configs.embeddings_dir` | `str` |

## Model Specific Parameters
| Config Field | CLI Option | Default Source | Type |
|-------------|------------|----------------|------|
| `factor` | `--factor` | `configs.factor` | `int` |
| `enc_in` | `--enc-in` | `configs.enc_in` | `int` |
| `freq` | `--freq` | `configs.freq` | `str` |

## Test Parameters
| Config Field | CLI Option | Default Source | Type |
|-------------|------------|----------------|------|
| `test_interval` | `--test-interval` | `configs.test_interval` | `int` |
| `test_iteration_interval` | `--test-iteration-interval` | `configs.test_iteration_interval` | `int` |

## Logging Parameters
| Config Field | CLI Option | Default Source | Type |
|-------------|------------|----------------|------|
| `log_every_n_iterations` | `--log-every-n-iterations` | `configs.log_every_n_iterations` | `int` |
| `save_iteration_metrics` | `--save-iteration-metrics/--no-save-iteration-metrics` | `configs.save_iteration_metrics` | `bool` |

## CLI-Only Options
These options don't have corresponding config fields and have their own defaults:

| CLI Option | Default Value | Type | Purpose |
|------------|---------------|------|---------|
| `--auto-start-tensorboard/--no-auto-start-tensorboard` | `True` | `bool` | Control TensorBoard auto-start |
| `--tensorboard-port` | `6006` | `int` | TensorBoard port |
| `--open-browser/--no-open-browser` | `False` | `bool` | Auto-open browser |
| `--resume-checkpoint` | `None` | `str` | Resume from checkpoint |
| `--quick-test` | `False` | `bool` | Quick test mode |
| `--extract-embeddings-only` | `False` | `bool` | Extract embeddings only |
| `--seed` | `2024` | `int` | Random seed |

## Runtime/Computed Fields (No CLI Options)
These config fields are computed at runtime and don't have CLI options:

- `available_files`
- `train_files`
- `test_files`
- `val_files`
- `train_data_path` (property)
- `val_data_path` (property)
- `test_data_path` (property)

## Special Value Handling

### Comma-Separated Lists
These CLI options are parsed as comma-separated strings and converted to lists:
- `--stocks` → `List[str]` (None means all stocks)
- `--features` → `List[str]`
- `--val-stocks` → `List[str]`
- `--device-ids` → `List[int]`

### JSON Parsing
- `--loss-kwargs`: Parsed as JSON string to dict

### Boolean Flags
Boolean options use the `--flag/--no-flag` syntax and show their defaults correctly.

## Adding New Configuration Options

To add a new configuration option:

1. **Add to `configs.py`**: Define the field in `StockPredictionConfig` with default value
2. **Add to `train.py`**: Add corresponding `@click.option` with `default=_config_defaults['field_name']` and `show_default=True`
3. **Test**: Run `python test_config_consistency.py` to verify consistency

The system will automatically handle the mapping between config field and CLI option.

## Consistency Verification

Run `python test_config_consistency.py` to verify:
1. All config fields (except runtime/computed ones) have CLI options
2. All CLI options correspond to config fields or are documented CLI-only options
3. Complete field coverage and consistency

## Example Usage

```bash
# Use all defaults from configs.py
python train.py

# Override specific config values
python train.py --seq-len 120 --batch-size 32 --learning-rate 1e-3

# See all defaults in help
python train.py --help
```

## Maintenance

- ✅ **Single Source**: Only edit defaults in `configs.py`
- ✅ **Automatic Sync**: CLI automatically inherits config defaults
- ✅ **Visible Help**: Users always see current defaults in CLI help
- ✅ **Type Safety**: Dataclass validation prevents configuration errors
