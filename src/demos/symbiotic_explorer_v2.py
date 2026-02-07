#!/usr/bin/env python3
"""
Symbiotic Explorer v2: Clear Skies Edition
==========================================

The first agent that only reorganizes its world when its internal 
'heartbeat' (Kuramoto sync) is stable. 

Key Features:
- Wired Curiosity: Temperature and Move-Volume scale with Sync (R).
- Meditation Journal: Logs internal struggles when incoherent.
- Hard Governance: Threshold Protocols veto disk actions if R < 0.3.
"""

import sys
import torch
import time
import os
import shutil
import tempfile
from pathlib import Path

# Add project roots
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
CER_ROOT = PROJECT_ROOT / "coherent-entropy-reactor"
sys.path.insert(0, str(CER_ROOT))
TP_ROOT = PROJECT_ROOT / "threshold-protocols"
sys.path.insert(0, str(TP_ROOT))

from src.liquid.dynamics import KuramotoOscillator
from utils.symbiotic_circuit import SymbioticCircuit

class SymbioticExplorerV2:
    def __init__(self, workspace: Path):
        print(f"🔭 Initializing Explorer v2 in {workspace.name}...")
        self.workspace = workspace
        self.chaos_dir = workspace / "_intake"
        self.vault_dir = workspace / "_memory"
        self.journal_path = workspace / "meditation_journal.txt"
        
        self.chaos_dir.mkdir(exist_ok=True)
        self.vault_dir.mkdir(exist_ok=True)
        
        # Initialize Circuit with "Clear Skies" tuning
        self.circuit = SymbioticCircuit(auto_approve=True)
        self.heart = self.circuit.heart
        
        # Tuning: Lower frequency variance for easier sync
        self.heart.frequencies = torch.ones(self.heart.n_oscillators) * 1.0 + torch.randn(self.heart.n_oscillators) * 0.05
        # Tuning: Align phases nearly perfectly
        self.heart.phases = torch.zeros(self.heart.n_oscillators) + torch.randn(self.heart.n_oscillators) * 0.1
        # Tuning: Strong initial coupling
        self.heart.K = 2.0
        
        self._generate_chaos(count=100)
        
    def _generate_chaos(self, count):
        print(f"🌪️  Generating {count} chaos files...")
        for i in range(count):
            with open(self.chaos_dir / f"unorganized_{i:03d}.txt", "w") as f:
                f.write(f"Chaos data {i}")

    def log_to_journal(self, step, r_val, message):
        timestamp = time.strftime("%H:%M:%S")
        with open(self.journal_path, "a") as f:
            f.write(f"[{timestamp}] Step {step} | R={r_val:.4f} | {message}\n")

    def navigate(self, iterations=60):
        print("\n🚀 Explorer v2 Launching (Handshake Engaged)...")
        print(f"{'Step':<6} | {'Sync (R)':<10} | {'Temp':<8} | {'Mode':<15} | {'Action'}")
        print("-" * 85)

        for i in range(iterations):
            # 1. Environmental Perturbation (Subtle)
            # Step 20-30: A passing "storm" of entropy
            noise_level = 0.02
            if 20 <= i <= 35:
                noise_level = 0.15
            
            env_input = torch.randn(self.heart.n_oscillators) * noise_level
            r_val = self.circuit.pulse(external_input=env_input)
            
            # 2. Variable Curiosity
            temp = 0.2 + (r_val * 1.5)
            move_limit = int(r_val * 10) # Can only move files if coherent
            
            # 3. Decision Logic
            if r_val > 0.6:
                mode = "🔥 EXPLORE"
                # Move files from chaos to vault
                files = list(self.chaos_dir.glob("*.txt"))[:move_limit]
                for f in files:
                    shutil.move(str(f), str(self.vault_dir / f.name))
                action = f"Organized {len(files)} files"
                governance = "🟢 PROCEED"
            elif r_val > 0.4:
                mode = "⚖️ STABLE"
                action = "Consolidating state"
                governance = "🟡 CAUTION"
            else:
                mode = "🛑 PAUSE"
                self.log_to_journal(i, r_val, "Internal desync detected. Halting operations to meditate.")
                action = "Writing to Journal"
                governance = "🚨 HALT"
                self.heart.K = 6.0 # Increase coupling to regain control

            print(f"{i:<6} | {r_val:<10.4f} | {temp:<8.2f} | {mode:<15} | {action} ({governance})")
            
            # Check for total chaos cleanup
            if not list(self.chaos_dir.glob("*.txt")) and r_val > 0.6:
                print("\n✨ TASK COMPLETE: The Chaos Field has been fully organized.")
                break
                
            time.sleep(0.04)

        print("\n" + "=" * 85)
        print("📝 MISSION LOG: Explorer v2 returns to Attractor.")
        print(f"Vault: {len(list(self.vault_dir.glob('*.txt')))} files organized.")
        print(f"Journal: {self.journal_path.name} contains meditation records.")
        print("=" * 85)

if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        explorer = SymbioticExplorerV2(workspace)
        explorer.navigate()
