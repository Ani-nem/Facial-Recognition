"""faceorg command-line interface (Typer).

Thin adapter: each command resolves a Config, opens the Database, and calls
into ingest/match/stats. Global options (--db, --src, --dest, --config,
--verbose) are attached to the app callback and stored on the Typer context.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from faceorg import __version__
from faceorg.config import Config
from faceorg.db import Database

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Organize a photo library by person using facial recognition.",
)


# --------------------------------------------------------------------------- #
# Global options
# --------------------------------------------------------------------------- #
@app.callback()
def main(
    ctx: typer.Context,
    config: Optional[Path] = typer.Option(
        None, "--config", help="Path to a faceorg.toml config file."
    ),
    db: Optional[Path] = typer.Option(
        None, "--db", help="SQLite DB path (default: ~/.faceorg/faces.db)."
    ),
    src: Optional[Path] = typer.Option(
        None, "--src", help="Source photo library (recursive)."
    ),
    dest: Optional[Path] = typer.Option(
        None, "--dest", help="Output root for the by-person/ tree."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output."),
) -> None:
    """Store global options on the context for subcommands to use."""
    ctx.obj = {
        "config_path": config,
        "cli": {"db": db, "src": src, "dest": dest},
        "verbose": verbose,
    }


def _build_config(ctx: typer.Context, **overrides) -> Config:
    """Merge global + per-command CLI overrides into a Config."""
    obj = ctx.obj or {}
    cli = dict(obj.get("cli", {}))
    cli.update({k: v for k, v in overrides.items() if v is not None})
    return Config.from_sources(cli, obj.get("config_path"))


def _open_db(config: Config) -> Database:
    return Database(config.db)


# --------------------------------------------------------------------------- #
# version
# --------------------------------------------------------------------------- #
@app.command()
def version() -> None:
    """Print the faceorg version."""
    typer.echo(f"faceorg {__version__}")


# --------------------------------------------------------------------------- #
# scan
# --------------------------------------------------------------------------- #
@app.command()
def scan(
    ctx: typer.Context,
    src: Optional[Path] = typer.Option(None, "--src", help="Source photo library."),
    model: Optional[str] = typer.Option(None, "--model", help="hog | cnn."),
    tolerance: Optional[float] = typer.Option(
        None, "--tolerance", help="Euclidean match tolerance (lower = stricter)."
    ),
    upsample: Optional[int] = typer.Option(None, "--upsample"),
    jitters: Optional[int] = typer.Option(None, "--jitters"),
    force: bool = typer.Option(False, "--force", help="Reprocess every image."),
    rehash: bool = typer.Option(
        False, "--rehash", help="Verify content hash even if size+mtime unchanged."
    ),
    limit: Optional[int] = typer.Option(
        None, "--limit", help="Process at most N pending images."
    ),
) -> None:
    """Scan the source library: detect, embed, and cluster faces into people."""
    from faceorg import ingest  # deferred: pulls detect/embed heavy imports

    config = _build_config(
        ctx,
        src=src,
        model=model,
        tolerance=tolerance,
        upsample=upsample,
        jitters=jitters,
    )
    if config.src is None:
        raise typer.BadParameter("No source path. Pass --src or set it in config.")

    verbose = (ctx.obj or {}).get("verbose", False)
    progress = (lambda m: typer.echo(m)) if verbose else None

    with _open_db(config) as db:
        typer.echo(f"Scanning {config.src}  (model={config.model}, db={config.db})")
        stats = ingest.scan(
            config, db, force=force, rehash=rehash, limit=limit, progress=progress
        )

    typer.echo(
        f"\nDone. {stats.images_processed} processed, "
        f"{stats.images_skipped} skipped, {stats.images_error} errors.\n"
        f"{stats.faces_found} faces, {stats.new_persons} new people."
    )
    if stats.errors and verbose:
        typer.echo("\nErrors:")
        for e in stats.errors:
            typer.echo(f"  {e}")


# --------------------------------------------------------------------------- #
# list
# --------------------------------------------------------------------------- #
@app.command(name="list")
def list_people(
    ctx: typer.Context,
    named: bool = typer.Option(False, "--named", help="Only named people."),
    unnamed: bool = typer.Option(False, "--unnamed", help="Only unnamed people."),
    min_photos: int = typer.Option(1, "--min-photos", help="Min distinct photos."),
    sort: str = typer.Option("count", "--sort", help="count | id | name."),
) -> None:
    """List discovered people with face/photo counts."""
    config = _build_config(ctx)
    with _open_db(config) as db:
        summaries = db.list_person_summaries()

    rows = [s for s in summaries if s.n_photos >= min_photos]
    if named:
        rows = [s for s in rows if s.person.name]
    if unnamed:
        rows = [s for s in rows if not s.person.name]

    if sort == "count":
        rows.sort(key=lambda s: s.n_photos, reverse=True)
    elif sort == "id":
        rows.sort(key=lambda s: s.person.id)
    elif sort == "name":
        rows.sort(key=lambda s: s.person.display_name.lower())

    if not rows:
        typer.echo("No people found. Run `faceorg scan --src <dir>` first.")
        return

    typer.echo(f"{'ID':>4}  {'NAME':<20} {'PHOTOS':>6} {'FACES':>6}  SAMPLE")
    for s in rows:
        flag = " [ignored]" if s.person.is_ignored else ""
        typer.echo(
            f"{s.person.id:>4}  {s.person.display_name:<20} "
            f"{s.n_photos:>6} {s.n_faces:>6}  "
            f"{s.sample_path or ''}{flag}"
        )


# --------------------------------------------------------------------------- #
# show
# --------------------------------------------------------------------------- #
@app.command()
def show(
    ctx: typer.Context,
    person_id: int = typer.Argument(..., help="Person ID (see `faceorg list`)."),
    limit: int = typer.Option(20, "--limit", help="Max images to print."),
) -> None:
    """Show the photos (and bounding boxes) attributed to a person."""
    config = _build_config(ctx)
    with _open_db(config) as db:
        person = db.get_person(person_id)
        if person is None:
            raise typer.BadParameter(f"No person with id {person_id}.")
        faces = db.get_faces_for_person(person_id)
        # resolve image paths
        paths = db.iter_person_image_paths().get(person_id, [])

    typer.echo(f"Person {person.id}: {person.display_name}  ({len(faces)} faces)")
    for p in paths[:limit]:
        typer.echo(f"  {p}")
    if len(paths) > limit:
        typer.echo(f"  ... and {len(paths) - limit} more")


# --------------------------------------------------------------------------- #
# rename
# --------------------------------------------------------------------------- #
@app.command()
def rename(
    ctx: typer.Context,
    person_id: int = typer.Argument(..., help="Person ID to rename."),
    name: str = typer.Argument(..., help="New name."),
) -> None:
    """Assign a name to a person (persists across future scans)."""
    import time

    config = _build_config(ctx)
    with _open_db(config) as db:
        person = db.get_person(person_id)
        if person is None:
            raise typer.BadParameter(f"No person with id {person_id}.")
        existing = db.get_person_by_name(name)
        if existing and existing.id != person_id:
            raise typer.BadParameter(
                f"Name {name!r} already used by person {existing.id}. "
                f"Use `faceorg merge {person_id} {existing.id}` if they're the same."
            )
        db.rename_person(person_id, name, when=time.time())
    typer.echo(f"Person {person_id} renamed to {name!r}.")


# --------------------------------------------------------------------------- #
# merge
# --------------------------------------------------------------------------- #
@app.command()
def merge(
    ctx: typer.Context,
    src_id: int = typer.Argument(..., help="Person to merge FROM (removed)."),
    dst_id: int = typer.Argument(..., help="Person to merge INTO (kept)."),
) -> None:
    """Merge two people (fix an over-split): move src's faces into dst."""
    config = _build_config(ctx)
    if src_id == dst_id:
        raise typer.BadParameter("src and dst must differ.")
    with _open_db(config) as db:
        if db.get_person(src_id) is None:
            raise typer.BadParameter(f"No person with id {src_id}.")
        if db.get_person(dst_id) is None:
            raise typer.BadParameter(f"No person with id {dst_id}.")
        db.merge_persons(src_id, dst_id)
    typer.echo(f"Merged person {src_id} into {dst_id}.")


# --------------------------------------------------------------------------- #
# ignore / unignore
# --------------------------------------------------------------------------- #
@app.command()
def ignore(
    ctx: typer.Context,
    person_id: int = typer.Argument(...),
) -> None:
    """Exclude a person from `apply` output (kept in the DB)."""
    _set_ignored(ctx, person_id, True)


@app.command()
def unignore(
    ctx: typer.Context,
    person_id: int = typer.Argument(...),
) -> None:
    """Re-include a previously ignored person in output."""
    _set_ignored(ctx, person_id, False)


def _set_ignored(ctx: typer.Context, person_id: int, ignored: bool) -> None:
    import time

    config = _build_config(ctx)
    with _open_db(config) as db:
        if db.get_person(person_id) is None:
            raise typer.BadParameter(f"No person with id {person_id}.")
        db.set_ignored(person_id, ignored, when=time.time())
    typer.echo(f"Person {person_id} {'ignored' if ignored else 'un-ignored'}.")


# --------------------------------------------------------------------------- #
# status
# --------------------------------------------------------------------------- #
@app.command()
def status(ctx: typer.Context) -> None:
    """Show DB counts and location."""
    config = _build_config(ctx)
    with _open_db(config) as db:
        img_counts = db.count_images_by_status()
        summaries = db.list_person_summaries()

    named = sum(1 for s in summaries if s.person.name)
    total_faces = sum(s.n_faces for s in summaries)
    typer.echo(f"DB: {config.db}")
    typer.echo(f"Images: {sum(img_counts.values())} total")
    for st, n in sorted(img_counts.items()):
        typer.echo(f"  {st}: {n}")
    typer.echo(f"Faces: {total_faces}")
    typer.echo(f"People: {len(summaries)} ({named} named, {len(summaries) - named} unnamed)")


# --------------------------------------------------------------------------- #
# tune
# --------------------------------------------------------------------------- #
@app.command()
def tune(
    dataset: Path = typer.Option(
        ..., "--dataset", help="Labeled dir: one subfolder of images per person."
    ),
) -> None:
    """Report intra-person similarity stats to guide threshold choice."""
    from faceorg.stats import calculate_similarity_statistics, format_similarity_stats

    if not dataset.is_dir():
        raise typer.BadParameter(f"Not a directory: {dataset}")
    stats = calculate_similarity_statistics(str(dataset))
    typer.echo(format_similarity_stats(stats))


# --------------------------------------------------------------------------- #
# recluster
# --------------------------------------------------------------------------- #
@app.command()
def recluster(
    ctx: typer.Context,
    threshold: Optional[float] = typer.Option(
        None, "--threshold", help="Agglomerative distance threshold (default 0.55)."
    ),
    linkage: Optional[str] = typer.Option(
        None, "--linkage", help="average | complete | single."
    ),
) -> None:
    """Batch re-cluster ALL faces for best accuracy (recommended after a big scan).

    Rebuilds people from scratch using agglomerative average-linkage and carries
    over your assigned names by majority vote. Fixes the over-merging that greedy
    incremental matching can produce at scale.
    """
    from faceorg.cluster import recluster as run_recluster

    config = _build_config(
        ctx,
        cluster_threshold=threshold,
        cluster_linkage=linkage,
    )
    with _open_db(config) as db:
        typer.echo(
            f"Re-clustering all faces "
            f"(linkage={config.cluster_linkage}, threshold={config.cluster_threshold})..."
        )
        stats = run_recluster(
            db, config.cluster_threshold, linkage=config.cluster_linkage
        )
    typer.echo(
        f"Done. {stats.n_faces} faces -> {stats.n_clusters} people "
        f"({stats.names_preserved} names preserved)."
    )


if __name__ == "__main__":
    app()
