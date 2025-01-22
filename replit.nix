{ pkgs }: {
  deps = [
    pkgs.python3
    pkgs.python3Packages.pip
  ];
  env = {
    PYTHONBIN = "${pkgs.python3}/bin/python";
    LANG = "en_US.UTF-8";
  };
}
