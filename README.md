# BOOTS-MBSAC

This is the implementation for the paper Bootstrapping the Expressivity with Model-based Planning. 

## Installment

```
# create a new environment if you want
# conda create -n <name> python=3.7
pip install -r requirements.txt
pip install -r lunzi/requirements.txt
```

## Run

To run a single experiment:

```bash
export PYTHONPATH=$PYTHONPATH:.
python examples/mb_sac.py --print_config --config configs/mbsac/walker2d.toml --log_dir /tmp/walker2d
```

You can find a TensorBoard event file at `/tmp/walker2d` and use the key `episode/boots/reward/mean` to generate your plot.  

You can also use `lunzi/run.py` to run many experiments together:

```
./lunzi/run.py --n_jobs 10 --template 'python examples/mb_sac.py --print_config --log_dir /tmp/walker2d --config configs/mbsac/walker2d.toml'
```

## License

MIT License. 
