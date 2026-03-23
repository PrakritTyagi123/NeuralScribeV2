; NeuralScribe v2 — Inno Setup Installer Script
; Download Inno Setup from https://jrsoftware.org/isdl.php
; Open this file in Inno Setup Compiler and click Build.

#define MyAppName "NeuralScribe"
#define MyAppVersion "2.0"
#define MyAppPublisher "Prathamesh Minde"
#define MyAppExeName "NeuralScribe.exe"

; IMPORTANT: Update this path to where Nuitka output is
#define BuildOutput "dist\launcher.dist"

[Setup]
AppId={{A1B2C3D4-E5F6-7890-ABCD-EF1234567890}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputBaseFilename=NeuralScribeSetup
OutputDir=installer_output
Compression=lzma2/ultra64
SolidCompression=yes
SetupIconFile=icon.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
PrivilegesRequired=admin
WizardStyle=modern
LicenseFile=LICENSE
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; All compiled files from Nuitka output
Source: "{#BuildOutput}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

; Empty data directories (user data — created fresh)
; These are handled in [Dirs] below

[Dirs]
Name: "{app}\datasets\english\cache"; Permissions: users-full
Name: "{app}\datasets\english\raw"; Permissions: users-full
Name: "{app}\models\english"; Permissions: users-full
Name: "{app}\models\english\exports"; Permissions: users-full
Name: "{app}\training_logs\english"; Permissions: users-full

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch NeuralScribe"; Flags: nowait postinstall skipifsilent
