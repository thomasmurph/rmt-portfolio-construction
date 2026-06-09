{
  description = "RMT covariance cleaning methods for minimum variance portfolio construction";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };

  outputs =
    { self, nixpkgs }:
    let
      pkgs = nixpkgs.legacyPackages.x86_64-linux;
    in
    {
      devShells.x86_64-linux.default = pkgs.mkShell {

        packages = [
          pkgs.python314
          pkgs.uv
        ];
        env = {
          UV_PYTHON_DOWNLOADS = "never";
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
            pkgs.stdenv.cc.cc.lib
            pkgs.zlib
          ];
        };
      };
    };
}
