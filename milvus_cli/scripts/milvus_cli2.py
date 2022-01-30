import sys
import os
import typing as t
from cmd import Cmd
import shlex

import rich
import typer
import tabulate

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from utils2 import PyOrm, Completer, getPackageVersion, WELCOME_MSG, EXIT_MSG
from Fs import readCsvFile
from Validation import (
    validateParamsByCustomFunc,
    validateCollectionParameter,
    validateIndexParameter,
    validateSearchParams,
    validateQueryParams,
    validateCalcParams,
)
from Types2 import ParameterException, ConnectException
from Types2 import MetricTypes, IndexTypesMap, IndexTypes, IndexParams


app = typer.Typer()
show_app = typer.Typer()
list_app = typer.Typer()
describe_app = typer.Typer()
create_app = typer.Typer()
app.add_typer(show_app, name="show")
app.add_typer(list_app, name="list")
app.add_typer(describe_app, name="describe")
app.add_typer(create_app, name="create")

def _get_target_collection(ctx: typer.Context, collection: str):
    ctx.obj.checkConnection()
    ctx.obj.getTargetCollection(collection)

@app.callback()
def setup(ctx: typer.Context):
    ctx.obj = ctx.obj if hasattr(ctx, "obj") and ctx.obj else PyOrm()

@app.command()
def clear():
    typer.clear()

@app.command()
def connect(ctx: typer.Context, alias: str = "default", host: str = "127.0.0.1", port: int = 19530, disconnect: bool = False):
    """
    Connect to Milvus.

    Example:

        milvus_cli > connect -h 127.0.0.1 -p 19530 -a default
    """
    try:
        ctx.obj.connect(alias, host, port, disconnect)
    except Exception as e:
        typer.echo(message=e, err=True)
    else:
        if disconnect:
            typer.echo("Disconnected.")
        else:
            typer.echo("Connect Milvus successfully.")
            typer.echo(ctx.obj.showConnection(alias))

def _load_release_fn(ctx: typer.Context, partition_fn: t.Callable, collection_fn: t.Callable, collection: str, partitions: t.Optional[t.List[str]] = None):
    try:
        _get_target_collection(ctx, collection)
        for partition in partitions:
            ctx.obj.getTargetPartition(collection, partition)
        result = partition_fn(collection, partitions) if partitions else collection_fn(collection)
    except Exception as e:
        typer.echo(message=e, err=True)
        return None
    else:
        return result

@app.command()
def load(ctx: typer.Context, collection: str, partitions: t.Optional[t.List[str]] = None):
    result = _load_release_fn(ctx, ctx.obj.loadPartitions, ctx.obj.loadCollection, collection, partitions)
    if result:
        if partitions:
            typer.echo(f"""Load {collection}'s partitions {partition} successfully""")
        else:
            typer.echo(f"""Load Collection {collection} successfully""")
        typer.echo(result)

@app.command()
def release(ctx: typer.Context, collection: str, partitions: t.Optional[t.List[str]] = None):
    result = _load_release_fn(ctx, ctx.obj.loadPartitions, ctx.obj.loadCollection, collection, partitions)
    if result:
        if partitions:
            typer.echo(f"""Release {collection}'s partitions {partition} successfully""")
        else:
            typer.echo(f"""Release Collection {collection} successfully""")
        typer.echo(result)

@show_app.command()
def connection(ctx: typer.Context, showAll: bool = False):
    try:
        (not showAll) and ctx.obj.checkConnection()
    except Exception as e:
        typer.echo("No connections.")
    else:
        typer.echo(ctx.obj.showConnection(showAll=showAll))

@show_app.command()
def loading_progress(ctx: typer.Context, collection: str, partitions: t.Optional[t.List[str]] = None):
    try:
        _get_target_collection(ctx, collection)
        result = ctx.obj.showCollectionLoadingProgress(collection, partitions)
    except Exception as e:
        typer.echo(message=e, err=True)
    else:
        typer.echo(
            tabulate(
                [[result.get("num_loaded_entities"), result.get("num_total_entities")]],
                headers=["num_loaded_entities", "num_total_entities"],
                tablefmt="pretty",
            )
        )

@show_app.command()
def index_progress(ctx: typer.Context, collection: str, index: str = ""):
    try:
        _get_target_collection(ctx, collection)
        result = ctx.obj.showIndexBuildingProgress(collection, index)
    except Exception as e:
        typer.echo(message=e, err=True)
    else:
        typer.echo(
            tabulate(
                [[result.get("indexed_rows"), result.get("total_rows")]],
                headers=["indexed_rows", "total_rows"],
                tablefmt="pretty",
            )
        )

@show_app.command()
def query_segment_info(ctx: typer.Context, collection: str, timeout: t.Optional[float] = None):
    typer.echo(ctx.obj.getQuerySegmentInfo(collection, timeout, prettierFormat=True))


@list_app.command()
def collections(ctx: typer.Context, timeout: t.Optional[float] = None, show_loaded: bool = False):
    try:
        ctx.obj.checkConnection()
        typer.echo(ctx.obj.listCollections(timeout, show_loaded))
    except Exception as e:
        typer.echo(message=e, err=True)

@list_app.command()
def partitions(ctx: typer.Context, collection: str):
    try:
        _get_target_collection(ctx, collection)
        typer.echo(ctx.obj.listPartitions(collection))
    except Exception as e:
        typer.echo(message=e, err=True)

@list_app.command("indexes")  # the plural of index is indices, but keeping this for consistency
def indices(ctx: typer.Context, collection: str):
    try:
        _get_target_collection(ctx, collection)
        typer.echo(ctx.obj.listIndexes(collection))
    except Exception as e:
        typer.echo(message=e, err=True)

@describe_app.command("collection")
def describe_collection(ctx: typer.Context, collection: str):
    try:
        ctx.obj.checkConnection()
        typer.echo(ctx.obj.getCollectionDetails(collection))
    except Exception as e:
        typer.echo(message=e, err=True)

@describe_app.command("partition")
def describe_partition(ctx: typer.Context, collection: str, partition: str = "_default"):
    try:
        ctx.obj.checkConnection()
        col = ctx.obj.getCollectionDetails(collection)
    except Exception as e:
        typer.echo(f"Error when getting collection ({collection}) by name", err=True)
    else:
        typer.echo(ctx.obj.getPartitionDetails(col, partition))

@describe_app.command("index")
def describe_index(ctx: typer.Context, collection: str):
    try:
        ctx.obj.checkConnection()
        col = ctx.obj.getCollectionDetails(collection)
    except Exception as e:
        typer.echo(f"Error when getting collection ({collection}) by name", err=True)
    else:
        typer.echo(ctx.obj.getIndexDetails(col))

@create_app.command("alias")
def create_alias(ctx: typer.Context, collection: str, alias_names: t.List[str], alter: bool = False, timeout: t.Optional[float] = None):
    try:
        ctx.obj.checkConnection()
        result = ctx.obj.alterCollectionAliasList(collection, alias_names, timeout) if alter else ctx.obj.createCollectionAliasList(collection, alias_names, timeout) 
    except ConnectException as ce:
        typer.echo(f"Error!\n{str(ce)}")
    else:
        if len(result) == len(alias_names):
            typer.echo(
                f"""{len(result)} alias {"altered" if alter else "created"} successfully."""
            )

@create_app.command("collection")
def create_collection(ctx: typer.Context, collection: str, primary_field: str, auto_id: bool = False, schema_description: str = "", fields: t.Optional[t.List[str]] = None): 
    # TODO: automatic validation of field types
    try:
        ctx.obj.checkConnection()
        validateCollectionParameter(collection, primary_field, fields)
    except (ParameterException, ConnectException) as e:
        typer.echo("Error!\n{str(e)}", err=True)
    else:
        typer.echo(
            ctx.obj.createCollection(
                collection, primary_field, auto_id, schema_description, fields
            )
        )
        typer.echo(f"Created collection ({collection}) successfully!")

@create_app.command("partition")
def create_partition(ctx: typer.Context, collection: str, partition: str, description: str = ""):
    try:
        ctx.obj.checkConnection()
        ctx.obj.getTargetCollection(collection)
    except Exception as e:
        typer.echo(f"Error when getting collection ({collection}) by name", err=True)
    else:
        typer.echo(ctx.obj.createPartition(collection, description, partition))
        typer.echo(f"Created partition ({partition}) successfully!")

def _ibp_callback(ctx: typer.Context, value: IndexParams):
    for param in ctx.params:
        if param.name == "index_type":
            break
    valid_build_params = IndexTypesMap[param.value].index_building_parameters
    if value not in valid_build_params:
        raise typer.BadParameter("{value} not valid. Must be one of {valid_build_params}.")
    return value

@create_app.command("index")
def create_index(ctx: typer.Context, collection: str, field: str, index_type: IndexTypes, metric_type: MetricTypes, index_building_parameters: t.List[IndexParams] = typer.Option(..., callback=_ibp_callback), timeout: t.Optional[float] = None):
    try:
        ctx.obj.checkConnection()
    except ConnectException as ce:
        typer.echo(f"Error\n{str(ce)}", err=True)
    else:
        typer.echo(
            ctx.obj.createIndex(collection, field, index_type, metric_type, index_building_parameters, timeout)
        )
        typer.echo("Created index successfully")


class Repl(Cmd):

    cli = typer.main.get_command(app)
    prompt = "milvus_cli > "
    intro = WELCOME_MSG
    
    @staticmethod
    def _catch_exit(cli, line):
        try:
            args = shlex.split(line)
            cli.main(args)
        except SystemExit:
            pass
        except Exception as e:
            raise e

    def default(self, line): 
        self._catch_exit(self.cli, line)

    def do_help(self, arg):
        self._catch_exit(self.cli, "--help")

    def do_exit(self, arg):
        typer.echo(EXIT_MSG)
        return True

def runRepl():
    repl = Repl()
    repl.cmdloop()

if __name__ == "__main__":
    runRepl()
