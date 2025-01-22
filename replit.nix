{ pkgs }: {
  deps = [
    pkgs.python39Full
    pkgs.python39Packages.pip
    pkgs.python39Packages.flask
    pkgs.replitPackages.stderred
  ];
  env = {
    PYTHON_LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
      pkgs.python39Full
    ];
    PYTHONBIN = "${pkgs.python39Full}/bin/python3.9";
    LANG = "en_US.UTF-8";
    STDERREDBIN = "${pkgs.replitPackages.stderred}/bin/stderred";
  };
}
