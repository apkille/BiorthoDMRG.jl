using Documenter
using BiorthoDMRG

makedocs(;
    sitename = "BiorthoDMRG.jl",
    modules = [BiorthoDMRG, BiorthoDMRG.Models],
    pages = ["Home" => "index.md"],
)

deploydocs(; repo = "github.com/apkille/BiorthoDMRG.jl", devbranch = "main")