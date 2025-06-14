#!/usr/bin/env python3
"""
Multi-Agent Research System CLI
Clean Click-based command line interface - uses existing project structure only
"""

import click
import subprocess
import sys
import time
import os
import json
import requests
from pathlib import Path
from typing import Optional, List

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# CLI Configuration
CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


def echo_success(message: str):
    """Print success message"""
    click.echo(click.style(f"✅ {message}", fg='green'))


def echo_error(message: str):
    """Print error message"""
    click.echo(click.style(f"❌ {message}", fg='red'))


def echo_warning(message: str):
    """Print warning message"""
    click.echo(click.style(f"⚠️ {message}", fg='yellow'))


def echo_info(message: str):
    """Print info message"""
    click.echo(click.style(f"ℹ️ {message}", fg='blue'))


def echo_step(step: str, message: str):
    """Print step message"""
    click.echo(click.style(f"{step} {message}", fg='white', bold=True))


def echo_header(message: str):
    """Print a header"""
    click.echo("\n" + "=" * 60)
    click.echo(click.style(message, fg='cyan', bold=True))
    click.echo("=" * 60)


class SystemManager:
    """Core system management functionality"""

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.compose_file = self.project_root / "docker-compose.yml"
        self.env_file = self.project_root / ".env"

    def check_prerequisites(self) -> bool:
        """Check system prerequisites"""
        echo_step("🔧", "Checking prerequisites...")

        # Check Docker
        if not self._check_command(["docker", "--version"]):
            echo_error("Docker not found. Please install Docker.")
            return False

        # Check Docker Compose
        if not self._check_command(["docker-compose", "--version"]):
            echo_error("Docker Compose not found. Please install Docker Compose.")
            return False

        # Check compose file
        if not self.compose_file.exists():
            echo_error(f"Docker compose file not found: {self.compose_file}")
            return False

        echo_success("All prerequisites met")
        return True

    def _check_command(self, cmd: List[str]) -> bool:
        """Check if a command exists"""
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def setup_environment(self) -> bool:
        """Setup environment file"""
        echo_step("📝", "Checking environment...")

        if not self.env_file.exists():
            env_defaults = self.project_root / ".env.defaults"
            if env_defaults.exists():
                with open(env_defaults, 'r') as src, open(self.env_file, 'w') as dst:
                    dst.write(src.read())
                echo_success("Created .env from .env.defaults")
            else:
                echo_error("No .env or .env.defaults found")
                return False
        else:
            echo_info(".env file exists")

        return True

    def check_sources(self):
        """Check sources directory - no creation, just report"""
        echo_step("📁", "Checking sources directory...")

        sources_dir = self.project_root / "sources"

        if not sources_dir.exists():
            echo_warning("Sources directory not found")
            echo_info("Create sources/ and add PDF files for document ingestion")
            return

        pdf_files = list(sources_dir.glob("*.pdf"))

        if pdf_files:
            echo_success(f"Found {len(pdf_files)} PDF files:")
            for i, pdf in enumerate(pdf_files[:5], 1):
                size_kb = pdf.stat().st_size // 1024
                click.echo(f"   {i}. {pdf.name} ({size_kb}KB)")
            if len(pdf_files) > 5:
                click.echo(f"   ... and {len(pdf_files) - 5} more")
        else:
            echo_info("No PDF files found in sources/")
            echo_info("Add PDF files to sources/ for document processing")

    def run_compose_command(self, cmd: List[str], timeout: int = 120) -> bool:
        """Run a docker-compose command"""
        try:
            full_cmd = ["docker-compose"] + cmd
            result = subprocess.run(
                full_cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            if result.returncode == 0:
                return True
            else:
                echo_error(f"Command failed: {' '.join(full_cmd)}")
                if result.stderr:
                    click.echo(f"Error: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            echo_error(f"Command timed out after {timeout}s")
            return False
        except Exception as e:
            echo_error(f"Command error: {e}")
            return False

    def wait_for_service(self, service: str, max_wait: int = 180) -> bool:
        """Wait for a service to become healthy"""
        echo_step("⏳", f"Waiting for {service} to be ready...")

        start_time = time.time()

        while time.time() - start_time < max_wait:
            try:
                result = subprocess.run(
                    ["docker-compose", "ps", service],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    timeout=10
                )

                if result.returncode == 0:
                    output = result.stdout.lower()
                    if "healthy" in output or "up" in output:
                        echo_success(f"{service} is ready")
                        return True

                elapsed = int(time.time() - start_time)
                if elapsed % 30 == 0 and elapsed > 0:
                    echo_info(f"Still waiting for {service}... ({elapsed}s/{max_wait}s)")

                time.sleep(5)

            except Exception as e:
                echo_warning(f"Error checking {service}: {e}")
                time.sleep(5)

        echo_error(f"Timeout waiting for {service}")
        return False

    def show_logs(self, service: Optional[str] = None, lines: int = 20):
        """Show service logs"""
        cmd = ["logs", "--tail", str(lines)]
        if service:
            cmd.append(service)

        try:
            full_cmd = ["docker-compose"] + cmd
            result = subprocess.run(
                full_cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                if service:
                    echo_step("📋", f"Recent {service} logs:")
                else:
                    echo_step("📋", "Recent system logs:")

                for line in result.stdout.split('\n'):
                    if line.strip():
                        click.echo(f"   {line}")
            else:
                echo_error("Failed to get logs")

        except Exception as e:
            echo_error(f"Error getting logs: {e}")


# Create the main CLI group
@click.group(context_settings=CONTEXT_SETTINGS)
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--project-dir', default='.', help='Project directory path')
@click.pass_context
def cli(ctx, verbose, project_dir):
    """🚀 Multi-Agent Research System CLI

    Command-line interface for managing your multi-agent research system.
    """
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    ctx.obj['manager'] = SystemManager(project_dir)


@cli.command()
@click.option('--timeout', '-t', default=600, help='Maximum startup timeout in seconds')
@click.option('--databases-only', is_flag=True, help='Start only database services')
@click.pass_context
def start(ctx, timeout, databases_only):
    """🚀 Start the multi-agent system"""

    manager: SystemManager = ctx.obj['manager']
    verbose = ctx.obj['verbose']

    echo_header("🚀 STARTING MULTI-AGENT RESEARCH SYSTEM")

    try:
        # Step 1: Prerequisites
        if not manager.check_prerequisites():
            raise click.ClickException("Prerequisites check failed")

        # Step 2: Environment setup
        if not manager.setup_environment():
            raise click.ClickException("Environment setup failed")

        # Step 3: Check sources (info only)
        manager.check_sources()

        # Step 4: Start services based on what's defined in docker-compose.yml
        echo_step("🐳", "Starting services...")

        if databases_only:
            # Start only database services
            db_services = ["neo4j", "mongodb", "chromadb"]
            if not manager.run_compose_command(["up", "-d"] + db_services):
                raise click.ClickException("Failed to start database services")
            echo_success("Database services started")

            # Wait for database health
            for service in db_services:
                if not manager.wait_for_service(service, max_wait=min(180, timeout // 3)):
                    echo_warning(f"Service {service} not ready, but continuing...")

            echo_success("Database services are running")
            return

        # Start all services defined in docker-compose.yml
        if not manager.run_compose_command(["up", "-d"]):
            raise click.ClickException("Failed to start services")

        echo_success("All services started")

        # Wait for key services (adjust based on your actual services)
        key_services = ["neo4j", "mongodb", "chromadb"]
        for service in key_services:
            if not manager.wait_for_service(service, max_wait=min(180, timeout // len(key_services))):
                echo_warning(f"Service {service} not ready, but continuing...")

        # Check if API service exists and wait for it
        result = subprocess.run(
            ["docker-compose", "config", "--services"],
            cwd=manager.project_root,
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            services = result.stdout.strip().split('\n')
            if "api" in services:
                echo_step("🚀", "Waiting for API service...")
                if not manager.wait_for_service("api", max_wait=120):
                    echo_warning("API service not ready, but may still work...")

        # Final verification
        echo_step("🔍", "Verifying system...")

        if verify_api_health(timeout=30):
            echo_success("API is responding correctly")
        else:
            echo_warning("API verification failed, but system may still work")

        # Show final status
        show_startup_summary()

        echo_header("✅ SYSTEM STARTUP COMPLETE")
        echo_success("Your multi-agent research system is ready!")

        # Show available commands
        click.echo("\n" + click.style("📋 Available Commands:", fg='cyan', bold=True))
        click.echo("   python cli.py status    # Check system status")
        click.echo("   python cli.py test      # Test system functionality")
        click.echo("   python cli.py logs      # View system logs")
        click.echo("   python cli.py health    # Run health checks")
        click.echo("   python cli.py stop      # Stop the system")

    except click.ClickException:
        raise
    except KeyboardInterrupt:
        echo_warning("Startup interrupted by user")
        ctx.exit(1)
    except Exception as e:
        echo_error(f"Unexpected error during startup: {e}")
        if verbose:
            import traceback
            click.echo(traceback.format_exc())
        ctx.exit(1)


@cli.command()
@click.option('--volumes', is_flag=True, help='Remove volumes too')
@click.confirmation_option(prompt='This will stop all services. Continue?')
@click.pass_context
def stop(ctx, volumes):
    """🛑 Stop all system services"""

    manager: SystemManager = ctx.obj['manager']

    echo_step("🛑", "Stopping multi-agent system...")

    cmd = ["down"]
    if volumes:
        cmd.append("-v")
        echo_warning("Removing volumes - all data will be lost!")

    if manager.run_compose_command(cmd):
        echo_success("System stopped successfully")

        if volumes:
            echo_info("All data volumes removed")
    else:
        echo_error("Failed to stop system")
        ctx.exit(1)


@cli.command()
@click.option('--service', '-s', help='Restart specific service only')
@click.pass_context
def restart(ctx, service):
    """🔄 Restart the system or specific service"""

    manager: SystemManager = ctx.obj['manager']

    if service:
        echo_step("🔄", f"Restarting {service}...")

        if manager.run_compose_command(["restart", service]):
            echo_success(f"{service} restarted successfully")
        else:
            echo_error(f"Failed to restart {service}")
            ctx.exit(1)
    else:
        echo_step("🔄", "Restarting entire system...")

        # Stop and start
        ctx.invoke(stop, volumes=False)
        time.sleep(3)
        ctx.invoke(start)


@cli.command()
@click.option('--format', 'output_format', type=click.Choice(['table', 'json']), default='table', help='Output format')
@click.pass_context
def status(ctx, output_format):
    """📊 Show system status"""

    manager: SystemManager = ctx.obj['manager']

    echo_step("📊", "Getting system status...")

    try:
        result = subprocess.run(
            ["docker-compose", "ps"],
            cwd=manager.project_root,
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            if output_format == 'json':
                # Simple status in JSON format
                status_data = {"status": "success", "output": result.stdout}
                click.echo(json.dumps(status_data, indent=2))
            else:
                echo_success("System status:")
                click.echo(result.stdout)
        else:
            echo_error("Failed to get system status")
            ctx.exit(1)

    except Exception as e:
        echo_error(f"Error getting status: {e}")
        ctx.exit(1)


@cli.command()
@click.option('--service', '-s', help='Show logs for specific service')
@click.option('--lines', '-n', default=50, help='Number of lines to show')
@click.option('--follow', '-f', is_flag=True, help='Follow log output')
@click.pass_context
def logs(ctx, service, lines, follow):
    """📋 Show service logs"""

    manager: SystemManager = ctx.obj['manager']

    if follow:
        echo_step("📋", f"Following logs{' for ' + service if service else ''}...")
        echo_info("Press Ctrl+C to stop")

        cmd = ["logs", "-f"]
        if service:
            cmd.append(service)

        try:
            subprocess.run(
                ["docker-compose"] + cmd,
                cwd=manager.project_root
            )
        except KeyboardInterrupt:
            echo_info("Log following stopped")
    else:
        manager.show_logs(service, lines)


@cli.command()
@click.option('--query', '-q', default='machine learning', help='Test query to send')
@click.option('--endpoint', default='agent', help='API endpoint to test')
@click.option('--timeout', default=60, help='Request timeout in seconds')
@click.option('--simple', '-s', is_flag=True, help='Show simple, clean output')
@click.pass_context
def test(ctx, query, endpoint, timeout, simple):
    """🧪 Test system functionality"""

    if simple:
        # Simple, clean output mode
        echo_step("🧪", f"Testing: '{query}'")

        try:
            url = f"http://localhost:8000/api/v1/{endpoint}"
            response = requests.post(url, json={"query": query}, timeout=timeout)

            if response.status_code == 200:
                result = response.json()
                message = result.get('message', '')
                system_health = result.get('system_health', {})

                # Clean output
                print(f"\n{'=' * 60}")
                print(f"📋 QUERY: {query}")
                print(f"{'=' * 60}")
                print(f"{message}")
                print(f"\n{'=' * 60}")
                print(f"🔧 SYSTEM STATUS:")
                for key, value in system_health.items():
                    clean_key = key.replace('_', ' ').title()
                    print(f"   {clean_key}: {value}")
                print(f"{'=' * 60}")

            else:
                echo_error(f"Request failed: {response.status_code}")

        except requests.exceptions.Timeout:
            echo_error(f"Request timed out after {timeout}s")
        except Exception as e:
            echo_error(f"Error: {e}")

        return

    # Original detailed output mode
    echo_step("🧪", f"Testing {endpoint} endpoint with query: '{query}'")

    try:
        url = f"http://localhost:8000/api/v1/{endpoint}"

        if endpoint == 'agent':
            response = requests.post(
                url,
                json={"query": query},
                timeout=timeout
            )
        else:
            response = requests.get(url, timeout=timeout)

        if response.status_code == 200:
            echo_success(f"API test successful!")

            try:
                result = response.json()

                if endpoint == 'agent':
                    message = result.get('message', '')
                    system_health = result.get('system_health', {})

                    # Clean metrics display
                    click.echo(f"\n📊 {click.style('Test Results:', fg='cyan', bold=True)}")
                    click.echo(f"   Query: {click.style(query, fg='white', bold=True)}")
                    click.echo(f"   Response Length: {click.style(f'{len(message):,} characters', fg='green')}")
                    click.echo(f"   Processing Time: {click.style(f'{timeout}s timeout', fg='blue')}")

                    # System health
                    click.echo(f"\n🔧 {click.style('System Health:', fg='cyan', bold=True)}")
                    for key, value in system_health.items():
                        clean_key = key.replace('_', ' ').title()
                        # Color code based on status
                        if '✅' in value:
                            color = 'green'
                        elif '🟡' in value:
                            color = 'yellow'
                        elif '❌' in value:
                            color = 'red'
                        else:
                            color = 'white'

                        click.echo(f"   {clean_key}: {click.style(value, fg=color)}")

                    # Show response preview
                    click.echo(f"\n📄 {click.style('Response Preview:', fg='cyan', bold=True)}")

                    # Show first 300 chars of clean message
                    preview = message[:300].replace('\n', ' ').strip()
                    if len(message) > 300:
                        preview += "..."
                    click.echo(f"   {preview}")

                else:
                    click.echo(f"\n📄 Response: {json.dumps(result, indent=2)}")

            except json.JSONDecodeError:
                click.echo(f"\n📄 Raw Response: {response.text}")

        else:
            echo_error(f"API test failed with status {response.status_code}")
            click.echo(f"Response: {response.text}")
            ctx.exit(1)

    except requests.exceptions.ConnectionError:
        echo_error("Cannot connect to API - is the system running?")
        echo_info("Try: python cli.py start")
        ctx.exit(1)
    except requests.exceptions.Timeout:
        echo_error(f"Request timed out after {timeout}s")
        ctx.exit(1)
    except Exception as e:
        echo_error(f"Test failed: {e}")
        ctx.exit(1)


@cli.command()
@click.option('--detailed', is_flag=True, help='Run detailed health checks')
@click.pass_context
def health(ctx, detailed):
    """🏥 Run system health checks"""

    echo_step("🏥", "Running system health checks...")

    health_results = []

    # Check API
    echo_info("Checking API health...")
    api_healthy = verify_api_health(timeout=10)
    health_results.append(("API", api_healthy))

    if api_healthy:
        echo_success("API is responding")
    else:
        echo_error("API is not responding")

    if detailed:
        # Check databases using existing project imports
        echo_info("Checking database connections...")

        # Neo4j
        try:
            from src.databases.graph.config import get_neo4j_driver
            driver = get_neo4j_driver()
            driver.verify_connectivity()

            with driver.session() as session:
                result = session.run("MATCH (n) RETURN count(n) as count")
                node_count = result.single()["count"]

            echo_success(f"Neo4j: Connected ({node_count} nodes)")
            health_results.append(("Neo4j", True))

        except Exception as e:
            echo_error(f"Neo4j: Failed ({e})")
            health_results.append(("Neo4j", False))

        # MongoDB
        try:
            from src.databases.document.config import get_mongodb_client, mongo_db_config
            client = get_mongodb_client()
            db = client[mongo_db_config.database]

            paper_count = db.papers.count_documents({})
            echo_success(f"MongoDB: Connected ({paper_count} papers)")
            health_results.append(("MongoDB", True))

        except Exception as e:
            echo_error(f"MongoDB: Failed ({e})")
            health_results.append(("MongoDB", False))

        # ChromaDB
        try:
            from src.databases.vector.config import ChromaDBConfig
            config = ChromaDBConfig()
            client = config.get_client()
            client.heartbeat()

            try:
                collection = client.get_collection("academic_papers")
                vector_count = collection.count()
                echo_success(f"ChromaDB: Connected ({vector_count} vectors)")
            except:
                echo_success("ChromaDB: Connected (no vectors)")

            health_results.append(("ChromaDB", True))

        except Exception as e:
            echo_error(f"ChromaDB: Failed ({e})")
            health_results.append(("ChromaDB", False))

    # Summary
    healthy_count = sum(1 for _, healthy in health_results if healthy)
    total_checks = len(health_results)

    click.echo(f"\n📊 Health Summary: {healthy_count}/{total_checks} checks passed")

    if healthy_count == total_checks:
        echo_success("System is fully healthy! 🎉")
    elif healthy_count > 0:
        echo_warning("System is partially healthy - some components need attention")
        ctx.exit(1)
    else:
        echo_error("System is unhealthy - major issues detected")
        ctx.exit(1)


# Helper functions
def verify_api_health(timeout: int = 30) -> bool:
    """Verify API is responding"""
    try:
        response = requests.get("http://localhost:8000/api/v1/status", timeout=timeout)
        return response.status_code == 200
    except:
        return False


def show_startup_summary():
    """Show startup summary"""
    echo_step("📋", "System Summary:")

    # Show access points
    click.echo("\n🌐 Access Points:")
    access_points = [
        ("API", "http://localhost:8000"),
        ("API Status", "http://localhost:8000/api/v1/status"),
        ("Neo4j Browser", "http://localhost:7474 (neo4j/password)"),
        ("MongoDB Express", "http://localhost:8081"),
        ("ChromaDB", "http://localhost:8001")
    ]

    for name, url in access_points:
        click.echo(f"   • {name}: {click.style(url, fg='blue')}")


if __name__ == '__main__':
    cli()