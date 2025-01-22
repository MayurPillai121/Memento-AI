{ pkgs }: {
  deps = [
    pkgs.python39Full
    pkgs.gcc
    pkgs.nodejs
    pkgs.poetry
  ];
  env = {
    PYTHONBIN = "${pkgs.python39Full}/bin/python3.9";
    LANG = "en_US.UTF-8";
    PIP_ROOT_USER_ACTION = "ignore";
    LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
      pkgs.python39Full
    ];
  };
}
