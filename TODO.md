# TODO List

- [x] Qualcosa che unifichi i test su tutti i modelli (unificare i main)
- [ ] RK implicito e testare RK esplicito con dt sufficientemente piccoli
- [x] Testare le estrapolazioni esplicite in convergenze spaziali
- [x] modificare e metodi di _ConvergenceTestPostprocessing per restituire c_h da salcare nell'output manager
- [ ] Print utility per le stampe di ConvergenceTestPostprocessing e SolvePostprocessing
- [ ] riempire le tabelle degli errori
- [x] aggiornare il readme 
- [x] pyptoject.toml
- [x] environment update
- [x] main per i plot sia dei grafici che delle mesh
- [ ] test di run con simulazione generica 
- [ ] test del plot custom
- [ ] decidere la struttura del foldr di configurazione(ad esempio deve contenere un unico fk_config.py ed eventualmente un .msh per generare la mesh, poi nel folder vengono generati gli xdmf/h5, i plot ecc, oppure si possono dividere le configurazioni in più scripts fk_run_config.py, fk_plot_config.py e ad esempio un results_config.py per tenere traccia degli errori)