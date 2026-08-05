"""User-management CLI: ``python -m api.users <command>``.

Operates on the same database the API uses (resolved from ``MORPH_API_*`` env vars
/ ``.env`` via ``api.runtime``). Passwords are argon2-hashed; setting a password
also invalidates that user's existing login tokens.

Examples:
    python -m api.users create alice              # prompts for a password
    python -m api.users passwd alice              # change a password
    python -m api.users list
    python -m api.users deactivate alice
    python -m api.users delete alice
"""
import click

from api.auth import service


def _fail(message: str) -> None:
    raise click.ClickException(message)


@click.group(help="Manage API users (create, edit, list, delete).")
def cli() -> None:
    pass


@cli.command(help="Create a new user.")
@click.argument("username")
@click.option("--password", help="Password (prompted securely if omitted).")
@click.option("--inactive", is_flag=True, help="Create the account deactivated.")
def create(username: str, password: str | None, inactive: bool) -> None:
    if not password:
        password = click.prompt("Password", hide_input=True, confirmation_prompt=True)
    try:
        user = service.create_user(username, password, is_active=not inactive)
    except service.AuthError as exc:
        _fail(str(exc))
    click.echo(f"Created user {user.username!r} (active={user.is_active}).")


@cli.command(help="Set a user's password (invalidates their existing tokens).")
@click.argument("username")
@click.option("--password", help="New password (prompted securely if omitted).")
def passwd(username: str, password: str | None) -> None:
    if not password:
        password = click.prompt("New password", hide_input=True, confirmation_prompt=True)
    try:
        service.set_password(username, password)
    except service.AuthError as exc:
        _fail(str(exc))
    click.echo(f"Updated password for {username!r}.")


@cli.command(name="list", help="List all users.")
def list_users() -> None:
    users = service.list_users()
    if not users:
        click.echo("No users.")
        return
    for user in users:
        click.echo(f"{user.username:<24} {'active' if user.is_active else 'inactive'}")


@cli.command(help="Reactivate a user.")
@click.argument("username")
def activate(username: str) -> None:
    try:
        service.set_active(username, True)
    except service.AuthError as exc:
        _fail(str(exc))
    click.echo(f"Activated {username!r}.")


@cli.command(help="Deactivate a user (blocks login; keeps the account).")
@click.argument("username")
def deactivate(username: str) -> None:
    try:
        service.set_active(username, False)
    except service.AuthError as exc:
        _fail(str(exc))
    click.echo(f"Deactivated {username!r}.")


@cli.command(help="Delete a user permanently.")
@click.argument("username")
@click.option("--yes", is_flag=True, help="Skip the confirmation prompt.")
def delete(username: str, yes: bool) -> None:
    if not yes:
        click.confirm(f"Delete user {username!r}?", abort=True)
    try:
        service.delete_user(username)
    except service.AuthError as exc:
        _fail(str(exc))
    click.echo(f"Deleted {username!r}.")


if __name__ == "__main__":
    cli()
