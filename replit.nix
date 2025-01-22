{ pkgs }: {
  deps = [
    pkgs.python312
    pkgs.replitPackages.prybar-python312
    pkgs.replitPackages.stderred
  ];
  env = {
    PYTHON_LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
      pkgs.python312
    ];
    PYTHONBIN = "${pkgs.python312}/bin/python3.12";
    LANG = "en_US.UTF-8";
    STDERREDBIN = "${pkgs.replitPackages.stderred}/bin/stderred";
    PRYBAR_PYTHON_BIN = "${pkgs.replitPackages.prybar-python312}/bin/prybar-python312";
  };
}
