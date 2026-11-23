{
  description = "ChatGeo-Magi evaluation plotting environment";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
      pythonEnvironment = pkgs.python3.withPackages (pythonPackages: [
        pythonPackages.numpy
        pythonPackages.matplotlib
        pythonPackages.requests
        pythonPackages.pillow
        pythonPackages.python-dateutil
        pythonPackages.python-dotenv
      ]);
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = [ pythonEnvironment ];
      };
    };
}
