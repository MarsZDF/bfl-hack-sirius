"""Example: Interactive Director Workflow.

This script demonstrates the "Director's Review" workflow:
1. Plan the transition (using Claude).
2. Review and edit the prompts.
3. Generate a quick preview (using Flux Schnell).
4. Render the final high-quality video (using Flux Pro).
"""

import json
from rich.console import Console
from rich.table import Table
from rich.prompt import Confirm, Prompt

from sirius import morph, plan_morph, TransitionStyle

console = Console()

def print_plan(plan):
    """Display the transition plan in a nice table."""
    table = Table(title="🎬 Transition Plan")
    table.add_column("Frame", style="cyan", justify="right")
    table.add_column("Prompt", style="white")

    for i, prompt in enumerate(plan.prompts):
        table.add_row(f"{i+1}", prompt)
    
    console.print(table)

def main():
    console.print("[bold green]Sirius Interactive Director[/bold green]")
    
    # 1. Setup
    image_a = "assets/cat.png"  # diligent-cat
    image_b = "assets/dog.png"  # diligent-dog
    # Ensure these exist or use placeholders if running strictly as demo
    
    # 2. Plan
    console.print(f"\n[bold yellow]Step 1: Planning transition...[/bold yellow]")
    try:
        plan = plan_morph(
            image_a, 
            image_b, 
            frame_count=8,  # Fewer frames for demo
            transition_style=TransitionStyle.MORPH
        )
    except Exception as e:
        console.print(f"[bold red]Error planning:[/bold red] {e}")
        return

    print_plan(plan)
    
    # 3. Edit (Simulated)
    if Confirm.ask("Would you like to edit any prompts?"):
        frame_idx = int(Prompt.ask("Enter frame number to edit", default="1")) - 1
        if 0 <= frame_idx < len(plan.prompts):
            new_prompt = Prompt.ask("New prompt", default=plan.prompts[frame_idx])
            plan.prompts[frame_idx] = new_prompt
            console.print("[green]Plan updated![/green]")
            print_plan(plan)

    # 4. Preview
    if Confirm.ask("Generate low-cost preview? (Flux Schnell)"):
        console.print("\n[bold yellow]Step 2: Generating preview...[/bold yellow]")
        result = morph(
            image_a, 
            image_b, 
            plan=plan, 
            preview=True,
            output_dir="outputs/preview"
        )
        console.print(f"[bold green]Preview saved to:[/bold green] {result.video_path}")
        
        # In a real GUI, we would show the image here.
        # For CLI, user has to open it.

    # 5. Final Render
    if Confirm.ask("Render final video? (Flux Pro)"):
        console.print("\n[bold yellow]Step 3: Rendering final video...[/bold yellow]")
        result = morph(
            image_a, 
            image_b, 
            plan=plan, 
            preview=False,
            output_dir="outputs/final"
        )
        console.print(f"[bold green]Video saved to:[/bold green] {result.video_path}")

if __name__ == "__main__":
    main()
