; Cappycat NSIS installer hooks (tauri.conf.json > bundle.windows.nsis.installerHooks).
;
; Uninstall removes the app (Tauri's own section deletes the installed files). When the user ticks
; the uninstaller's "Delete the application data" box it also removes %LOCALAPPDATA%\Cappycat -
; the managed Python environment, the AI models, the downloaded ffmpeg / uv, caches, logs and
; autosaves (docs/FEATURES_V2.md section 9). Documents\Cappycat (clips, projects, exports,
; characters, presets) is never touched.

!macro NSIS_HOOK_POSTUNINSTALL
  ${If} $DeleteAppDataCheckboxState = 1
  ${AndIf} $UpdateMode <> 1
    SetShellVarContext current
    RMDir /r "$LOCALAPPDATA\Cappycat\python"
    RMDir /r "$LOCALAPPDATA\Cappycat\models"
    RMDir /r "$LOCALAPPDATA\Cappycat\ffmpeg"
    RMDir /r "$LOCALAPPDATA\Cappycat\bin"
    RMDir /r "$LOCALAPPDATA\Cappycat\uv"
    RMDir /r "$LOCALAPPDATA\Cappycat\downloads"
    RMDir /r "$LOCALAPPDATA\Cappycat\cache"
    RMDir /r "$LOCALAPPDATA\Cappycat\logs"
    RMDir /r "$LOCALAPPDATA\Cappycat\autosave"
    RMDir "$LOCALAPPDATA\Cappycat"
  ${EndIf}
!macroend
