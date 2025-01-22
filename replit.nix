{ pkgs }: {
  deps = [
    pkgs.python310
    pkgs.python310Packages.pip
  ];
  env = {
    PYTHONPATH = "${pkgs.python310}/lib/python3.10/site-packages";
    LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
      pkgs.zlib
    ];
  };
}
