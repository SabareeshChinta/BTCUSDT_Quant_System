import json

def generate_report():
    with open("research_report_raw.json", "r") as f:
        data = json.load(f)
        
    print("# Final Quantitative Research Report\n")
    
    # 1. Base Strategy vs ML Strategy Walk-Forward Analysis
    print("## Walk-Forward Analysis\n")
    for tf, windows in data["WalkForward"].items():
        print(f"### Timeframe: {tf}")
        
        base_wins = 0
        base_losses = 0
        base_sqns = []
        ml_wins = 0
        ml_losses = 0
        ml_sqns = []
        
        for w in windows:
            print(f"**Test Window: {w['Test']}** (Train: {w['Train']})")
            
            b = w["Base"]
            print(f"- **Base**: Trades: {b['Trades']}, Win Rate: {b['WinRate']:.1%}, Net PnL: ${b['NetPnL']:.2f}, Max DD: {b['MaxDD']:.2f}%, SQN: {b['SQN']:.2f}")
            if b['NetPnL'] > 0: base_wins += 1
            else: base_losses += 1
            base_sqns.append(b['SQN'])
            
            m = w["ML"]
            if m:
                print(f"- **ML**: Trades: {m['Trades']}, Win Rate: {m['WinRate']:.1%}, Net PnL: ${m['NetPnL']:.2f}, Max DD: {m['MaxDD']:.2f}%, SQN: {m['SQN']:.2f}")
                if m['NetPnL'] > 0: ml_wins += 1
                else: ml_losses += 1
                ml_sqns.append(m['SQN'])
            else:
                print("- **ML**: No trades or insufficient data.")
        print("")
        
        # Summary
        avg_base_sqn = sum(base_sqns)/len(base_sqns) if base_sqns else 0
        avg_ml_sqn = sum(ml_sqns)/len(ml_sqns) if ml_sqns else 0
        print(f"**{tf} Summary:**")
        print(f"- Base Strategy Profitable Windows: {base_wins}/{base_wins+base_losses} (Avg SQN: {avg_base_sqn:.2f})")
        print(f"- ML Strategy Profitable Windows: {ml_wins}/{ml_wins+ml_losses} (Avg SQN: {avg_ml_sqn:.2f})\n")

    # 2. Robustness Testing
    print("## Robustness Stress Test (Last Window)\n")
    for tf, scenarios in data.get("Robustness", {}).items():
        if not scenarios: continue
        print(f"### {tf} Stress Tests")
        print(f"- **High Slippage (2x)**: Win Rate: {scenarios['High_Slippage']['Win Rate']:.1%}, Net PnL: ${scenarios['High_Slippage']['Net PnL']:.2f}")
        print(f"- **High Fees (2x)**: Win Rate: {scenarios['High_Fees']['Win Rate']:.1%}, Net PnL: ${scenarios['High_Fees']['Net PnL']:.2f}\n")
        
    # 3. Parameter Stability
    print("## Parameter Stability Grid\n")
    for tf, results in data.get("ParameterStability", {}).items():
        if not results: continue
        
        profitable = sum(1 for r in results if r['net_pnl'] > 0)
        total = len(results)
        print(f"### {tf} Grid Search")
        print(f"- Tested {total} parameter combinations.")
        print(f"- Profitable combinations: {profitable} ({profitable/total:.1%})")
        print("- Stability Verdict: " + ("Highly Stable" if profitable/total > 0.8 else "Unstable/Fragile"))
        
if __name__ == "__main__":
    generate_report()
