def fig_paths(path, prefix="./out/", exts=[".pdf", ".png"]):
    return ["".join([prefix + path, e]) for e in exts]
