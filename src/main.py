#!/usr/bin/env python
import os
import sys
import json
from datetime import datetime

def main():
    print("=== BDA-DBC Experimental Framework ===")
    print(f"Starting experiments at {datetime.now()}")
    
    from train import experiment_domain_shift
    from evaluate import experiment_online_update  
    from preprocess import experiment_ablation_study
    
    os.makedirs(".research/iteration1/images", exist_ok=True)
    
    original_dir = os.getcwd()
    os.chdir(".research/iteration1/images")
    
    try:
        print("\n" + "="*50)
        experiment_domain_shift()
        
        print("\n" + "="*50) 
        experiment_online_update()
        
        print("\n" + "="*50)
        experiment_ablation_study()
        
        print("\n" + "="*50)
        print("All experiments completed successfully!")
        
        status_data = {
            "status_enum": "stopped",
            "completion_time": datetime.now().isoformat(),
            "experiments_completed": [
                "domain_shift_detection",
                "online_bayesian_update_efficiency", 
                "ablation_study"
            ]
        }
        
        os.chdir(original_dir)
        with open(".research/research_history.json", "w") as f:
            json.dump(status_data, f, indent=2)
            
        print(f"Status set to 'stopped' and saved to research_history.json")
        
    except Exception as e:
        os.chdir(original_dir)
        print(f"Error during experiment execution: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
