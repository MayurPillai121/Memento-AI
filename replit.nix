{ pkgs }: {
  deps = [
    pkgs.python3
  ];
  env = {
    PYTHONBIN = "${pkgs.python3}/bin/python3";
    LANG = "en_US.UTF-8";
  };
}
