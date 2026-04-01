# src/utils/Progress.jl
"""
Common utility functions for progress reporting across PME modules.
Consolidates duplicate progress bar implementations.
"""
module ProgressUtils

using Printf

export print_progress_bar

"""
    print_progress_bar(current, total, label="Progress")

Renders a simple text-based progress bar in the console.
This is the consolidated implementation used across all PME modules.

# Arguments
- `current::Int`: Current progress value
- `total::Int`: Total/maximum progress value  
- `label::String`: Label to display with the progress bar

# Example
```julia
for i in 1:100
    print_progress_bar(i, 100, "Processing")
    # ... do work ...
end
```
"""
function print_progress_bar(current, total, label="Progress")
    percentage = current / total * 100
    bar_length = 40
    filled_length = round(Int, bar_length * current / total)
    
    # Construct the bar string
    bar = "█"^filled_length * "░"^(bar_length - filled_length)
    
    # Print the bar, overwriting the previous line
    @printf("\r  %s: [%s] %5.1f%% (%d/%d)", label, bar, percentage, current, total)
    flush(stdout)
end

end # module ProgressUtils
