import click

@click.command()
@click.option('--params', default='', help='Path to parameter file.')
@click.argument('i', nargs=-1,type=click.Path(exists=True))

def main(params, i):
    if params:
        params_dict = {}
        with open(params) as f:
            line = f.readline()
            while line:
                param_name, param_value = line.split(',')
                params_dict[param_name] = param_value
                line = f.readline()

    for ix in i:
        print ix
