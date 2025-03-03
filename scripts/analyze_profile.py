"""
Script to analyze profile data and print useful statistics.

Usage:
    python scripts/analyze_profile.py <profile_file>
    
Example:
    python scripts/analyze_profile.py profiling/test_tensor.prof
    
This script will:
1. Display overall profile statistics
2. Show top functions by cumulative time
3. Show top functions by total time
4. Show top tensor.py functions
5. Show top NetworkX functions
"""
import pstats
import sys
from pstats import SortKey

def analyze_profile(profile_path):
    """Analyze a profiling data file and print useful statistics."""
    stats = pstats.Stats(profile_path)
    
    print(f"\n{'='*80}")
    print(f"Profile Analysis for: {profile_path}")
    print(f"{'='*80}\n")
    
    # Print overall statistics
    print("Overall Statistics:")
    print(f"Total calls: {stats.total_calls}")
    print(f"Total primitive calls: {stats.prim_calls}")
    print(f"Total time: {stats.total_tt:.4f} seconds")
    print()
    
    # Print top functions by cumulative time
    print("Top 20 Functions (by cumulative time):")
    stats.sort_stats(SortKey.CUMULATIVE).print_stats(20)
    print()
    
    # Print top functions by total time
    print("Top 20 Functions (by total time):")
    stats.sort_stats(SortKey.TIME).print_stats(20)
    print()
    
    # Print specific function groups
    
    # Find tensor.py functions
    tensor_funcs = []
    for func in stats.stats:
        if func[0] is not None and '/tensor.py' in func[0]:
            tensor_funcs.append((func, stats.stats[func]))
    
    # Find networkx functions
    nx_funcs = []
    for func in stats.stats:
        if func[0] is not None and 'networkx' in func[0]:
            nx_funcs.append((func, stats.stats[func]))
    
    # Sort by cumulative time
    tensor_funcs.sort(key=lambda x: x[1][3], reverse=True)
    nx_funcs.sort(key=lambda x: x[1][3], reverse=True)
    
    # Print results
    print("Top tensor.py Functions:")
    for i, (func, stat) in enumerate(tensor_funcs[:10]):
        ncalls, tottime, p_calls, cumtime, callers = stat
        print(f"{i+1:2d}. {func[2]:30} {cumtime:8.4f}s  (calls: {ncalls})")
    
    print("\nTop NetworkX Functions:")
    for i, (func, stat) in enumerate(nx_funcs[:10]):
        ncalls, tottime, p_calls, cumtime, callers = stat
        print(f"{i+1:2d}. {func[2]:30} {cumtime:8.4f}s  (calls: {ncalls})")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/analyze_profile.py <profile_file>")
        sys.exit(1)
    
    analyze_profile(sys.argv[1])