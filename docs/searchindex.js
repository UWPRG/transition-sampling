Search.setIndex({docnames:["engines/cp2k","engines/engines","engines/gromacs","engines/new_engine","index","input/aimless","input/colvar","input/input","input/likelihood","modules","transition_sampling","transition_sampling.algo","transition_sampling.colvar","transition_sampling.engines","transition_sampling.engines.cp2k","transition_sampling.engines.gromacs","transition_sampling.likelihood","transition_sampling.util","transition_sampling.util.atomic_masses","workflow/aimless","workflow/colvar","workflow/likelihood","workflow/workflow"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":3,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,sphinx:56},filenames:["engines/cp2k.rst","engines/engines.rst","engines/gromacs.rst","engines/new_engine.rst","index.rst","input/aimless.rst","input/colvar.rst","input/input.rst","input/likelihood.rst","modules.rst","transition_sampling.rst","transition_sampling.algo.rst","transition_sampling.colvar.rst","transition_sampling.engines.rst","transition_sampling.engines.cp2k.rst","transition_sampling.engines.gromacs.rst","transition_sampling.likelihood.rst","transition_sampling.util.rst","transition_sampling.util.atomic_masses.rst","workflow/aimless.rst","workflow/colvar.rst","workflow/likelihood.rst","workflow/workflow.rst"],objects:{"":{transition_sampling:[10,0,0,"-"]},"transition_sampling.algo":{acceptors:[11,0,0,"-"],aimless_shooting:[11,0,0,"-"]},"transition_sampling.algo.acceptors":{AbstractAcceptor:[11,1,1,""],DefaultAcceptor:[11,1,1,""],MultiBasinAcceptor:[11,1,1,""]},"transition_sampling.algo.acceptors.AbstractAcceptor":{is_accepted:[11,2,1,""]},"transition_sampling.algo.acceptors.DefaultAcceptor":{is_accepted:[11,2,1,""]},"transition_sampling.algo.acceptors.MultiBasinAcceptor":{is_accepted:[11,2,1,""],products:[11,3,1,""],reactants:[11,3,1,""]},"transition_sampling.algo.aimless_shooting":{AimlessShootingDriver:[11,1,1,""],AsyncAimlessShooting:[11,1,1,""],ResultsLogger:[11,1,1,""],generate_velocities:[11,5,1,""]},"transition_sampling.algo.aimless_shooting.AimlessShootingDriver":{base_acceptor:[11,3,1,""],base_engine:[11,3,1,""],log_name:[11,3,1,""],position_dir:[11,3,1,""],run:[11,2,1,""],temp:[11,3,1,""]},"transition_sampling.algo.aimless_shooting.AsyncAimlessShooting":{accepted_states:[11,3,1,""],acceptor:[11,3,1,""],current_offset:[11,3,1,""],current_start:[11,3,1,""],engine:[11,3,1,""],is_accepted:[11,2,1,""],pick_starting:[11,2,1,""],position_dir:[11,3,1,""],results_logger:[11,3,1,""],run:[11,2,1,""],temp:[11,3,1,""]},"transition_sampling.algo.aimless_shooting.ResultsLogger":{base_logger:[11,3,1,""],csv_name:[11,4,1,""],cur_index:[11,3,1,""],log_result:[11,2,1,""],name:[11,3,1,""],xyz_name:[11,4,1,""]},"transition_sampling.colvar":{plumed_driver:[12,0,0,"-"]},"transition_sampling.colvar.plumed_driver":{PlumedDriver:[12,1,1,""]},"transition_sampling.colvar.plumed_driver.PlumedDriver":{run:[12,2,1,""]},"transition_sampling.driver":{execute:[10,5,1,""],main:[10,5,1,""],parse_aimless:[10,5,1,""],parse_colvar:[10,5,1,""],parse_engine:[10,5,1,""],parse_likelihood:[10,5,1,""],read_and_run:[10,5,1,""],run_aimless:[10,5,1,""],run_colvar:[10,5,1,""],run_likelihood:[10,5,1,""]},"transition_sampling.engines":{abstract_engine:[13,0,0,"-"],cp2k:[14,0,0,"-"],gromacs:[15,0,0,"-"],plumed:[13,0,0,"-"]},"transition_sampling.engines.abstract_engine":{AbstractEngine:[13,1,1,""],ShootingResult:[13,1,1,""]},"transition_sampling.engines.abstract_engine.AbstractEngine":{atoms:[13,4,1,""],box_size:[13,4,1,""],flip_velocity:[13,2,1,""],get_engine_str:[13,2,1,""],md_cmd:[13,3,1,""],plumed_handler:[13,3,1,""],run_shooting_point:[13,2,1,""],set_delta_t:[13,2,1,""],set_positions:[13,2,1,""],set_velocities:[13,2,1,""],validate_inputs:[13,2,1,""]},"transition_sampling.engines.abstract_engine.ShootingResult":{fwd:[13,3,1,""],rev:[13,3,1,""]},"transition_sampling.engines.cp2k":{CP2K_engine:[14,0,0,"-"],CP2K_inputs:[14,0,0,"-"],CP2K_outputs:[14,0,0,"-"]},"transition_sampling.engines.cp2k.CP2K_engine":{CP2KEngine:[14,1,1,""]},"transition_sampling.engines.cp2k.CP2K_engine.CP2KEngine":{atoms:[14,4,1,""],box_size:[14,4,1,""],cp2k_inputs:[14,3,1,""],flip_velocity:[14,2,1,""],get_engine_str:[14,2,1,""],set_delta_t:[14,2,1,""],set_positions:[14,2,1,""],set_velocities:[14,2,1,""],validate_inputs:[14,2,1,""]},"transition_sampling.engines.cp2k.CP2K_inputs":{CP2KInputsHandler:[14,1,1,""]},"transition_sampling.engines.cp2k.CP2K_inputs.CP2KInputsHandler":{atoms:[14,4,1,""],box_size:[14,4,1,""],cp2k_dict:[14,3,1,""],flip_velocity:[14,2,1,""],read_timestep:[14,2,1,""],set_plumed_file:[14,2,1,""],set_positions:[14,2,1,""],set_project_name:[14,2,1,""],set_traj_print_file:[14,2,1,""],set_traj_print_freq:[14,2,1,""],set_velocities:[14,2,1,""],temp:[14,4,1,""],write_cp2k_inputs:[14,2,1,""]},"transition_sampling.engines.cp2k.CP2K_outputs":{CP2KOutputHandler:[14,1,1,""]},"transition_sampling.engines.cp2k.CP2K_outputs.CP2KOutputHandler":{check_warnings:[14,2,1,""],copy_out_file:[14,2,1,""],get_out_file:[14,2,1,""],read_frames_2_3:[14,2,1,""]},"transition_sampling.engines.gromacs":{mdp:[15,0,0,"-"]},"transition_sampling.engines.gromacs.mdp":{MDPHandler:[15,1,1,""]},"transition_sampling.engines.gromacs.mdp.MDPHandler":{print_freq:[15,3,1,""],raw_string:[15,3,1,""],set_traj_print_freq:[15,2,1,""],timestep:[15,4,1,""],write_mdp:[15,2,1,""]},"transition_sampling.engines.plumed":{PlumedInputHandler:[13,1,1,""],PlumedOutputHandler:[13,1,1,""]},"transition_sampling.engines.plumed.PlumedInputHandler":{after:[13,3,1,""],before:[13,3,1,""],write_plumed:[13,2,1,""]},"transition_sampling.engines.plumed.PlumedOutputHandler":{check_basin:[13,2,1,""]},"transition_sampling.likelihood":{likelihood_max:[16,0,0,"-"],optimization:[16,0,0,"-"]},"transition_sampling.likelihood.likelihood_max":{Maximizer:[16,1,1,""],MaximizerSolution:[16,1,1,""],SingleSolution:[16,1,1,""]},"transition_sampling.likelihood.likelihood_max.Maximizer":{colvars:[16,3,1,""],maximize:[16,2,1,""],metadata:[16,3,1,""],niter:[16,3,1,""],use_jac:[16,3,1,""]},"transition_sampling.likelihood.likelihood_max.MaximizerSolution":{combinations:[16,3,1,""],max:[16,3,1,""],req_improvement:[16,3,1,""],to_csv:[16,2,1,""],to_dataframe:[16,2,1,""]},"transition_sampling.likelihood.optimization":{calc_p:[16,5,1,""],calc_r:[16,5,1,""],obj_func:[16,5,1,""],optimize:[16,5,1,""]},"transition_sampling.util":{atomic_masses:[18,0,0,"-"],periodic_table:[17,0,0,"-"],xyz:[17,0,0,"-"]},"transition_sampling.util.atomic_masses":{get_atomic_mass_dict:[18,5,1,""]},"transition_sampling.util.periodic_table":{atomic_symbols_to_mass:[17,5,1,""]},"transition_sampling.util.xyz":{read_xyz_file:[17,5,1,""],read_xyz_frame:[17,5,1,""],write_xyz_frame:[17,5,1,""]},transition_sampling:{algo:[11,0,0,"-"],colvar:[12,0,0,"-"],driver:[10,0,0,"-"],engines:[13,0,0,"-"],likelihood:[16,0,0,"-"],util:[17,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","attribute","Python attribute"],"4":["py","property","Python property"],"5":["py","function","Python function"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:attribute","4":"py:property","5":"py:function"},terms:{"0":[1,11,13,14,15,16,19,21,22],"1":[1,7,8,11,13,14,16,19,21,22],"10":[19,22],"100":[1,7,8,16],"1063":[19,22],"15":1,"15nm":1,"1nm":1,"1st":17,"2":[1,5,7,11,13,14,16,19,21,22],"2234477":[19,22],"3":[11,13,14,17,19,22],"3b":11,"3d":17,"4":[19,22],"4th":[19,22],"5":[21,22],"8":[19,22],"\u00e5":[3,5,7,19,22],"abstract":[11,13],"boolean":16,"case":[2,3,5,7,16,19,22],"class":[11,12,13,14,15,16],"default":[1,5,7,8,13,16,17],"do":[0,2,3,14,16,19,22],"final":[1,3,21,22],"float":[5,7,11,13,14,16,17,18],"function":[11,16,21,22],"int":[5,7,8,11,13,14,15,16],"long":[11,16],"new":[1,2,5,6,7,11,13,14,15,16,19,22],"null":[5,6,7,8],"public":3,"return":[10,11,12,13,14,15,16,17,18],"static":[3,11],"true":[7,8,11,13,14,16,17],"try":[11,19,22],"while":[3,11,16,19,22],A:[3,11,12,13,14,16,19,22],At:[0,2,19,22],For:[2,3,11,19,21,22],If:[0,2,3,5,6,7,8,10,11,13,14,15,16,17,19,21,22],In:[0,1,2,3,5,7,14,16,19,22],It:[3,7,15,19,21,22],One:[19,22],The:[0,1,2,3,5,7,10,11,13,14,15,16,19,21,22],Then:[3,21,22],There:[11,19,22],These:[2,3,5,6,7,8,11,13,19,22],To:[3,11,21,22],Will:[13,15,16],With:[3,21,22],_:[],_fatal:[0,2],abc:[11,13],abl:[3,5,7],about:[0,3,13,19,22],abov:[3,10,19,21,22],abstract_engin:[10,14],abstractacceptor:11,abstractengin:[3,10,11,13,14],accept:[5,7,11,16,19,21,22],accepted_st:11,acceptor:[5,7,10],access:[5,7],across:1,actual:13,ad:[3,12,16,22],add:[13,16],addit:[1,11,13,14,16,19,22],addition:[0,2],adjust:13,after:[1,3,5,7,8,11,13,15,19,22],afterward:[5,7],again:[19,20,22],aimless:[0,1,2,3,5,6,7,8,10,11,12,13,16,20,21],aimless_driv:7,aimless_input:[5,7,10],aimless_shoot:10,aimlessshoot:11,aimlessshootingdriv:[10,11],algo:10,algorithm:[1,11,13,22],all:[0,1,2,3,5,7,8,11,12,13,14,16,19,20],alloc:16,allow:[3,11,15,19,22],along:[19,22],alpha0:[16,21,22],alpha:[16,21,22],alpha_0:[16,21,22],alpha_ix_i:[16,21,22],alphas_1:16,alphas_m:16,alreadi:[3,11,16,19,22],also:[1,11,14,15,16,19,22],alter:14,amu:17,an:[0,1,3,5,7,11,13,14,15,16,17,19,21,22],analyt:[7,8,16],analyz:16,ani:[0,1,2,3,5,7,8,12,14,15,19,20,22],anoth:[11,16,19,22],anyth:[3,5,7,14,15],apart:1,append:[11,14,16,19,22],appropri:3,approxim:[7,8,16],ar:[0,1,2,3,5,6,7,8,10,11,13,14,16,19,21,22],aren:16,arg:[1,13,21,22],argon:[3,11],argument:[1,11,13],around:3,arrai:[3,11,13,14,16,17],assign:13,associ:[1,14],async:[11,13],asyncaimlessshoot:11,asynchron:3,atom:[0,1,2,11,13,14,17,18],atomic_mass:[10,17],atomic_symbols_to_mass:17,attach:[11,13],attempt:[19,21,22],attribut:13,automat:[2,12,14],avail:[2,21,22],await:[3,13],backward:1,bar:0,base:[2,3,11,12,13,14,15,16,22],base_acceptor:11,base_engin:11,base_logg:11,basin:[1,3,5,7,11,13,21],basin_ll1:1,basin_ll2:1,basin_ul1:1,basin_ul2:1,basinhop:[7,8,16],becaus:[19,22],becom:16,been:[3,14,15,17,19,22],befor:[2,3,5,7,11,13,14,16,19,22],being:[0,3,11,16],believ:[19,22],bell:[16,21,22],below:[0,1,2,5,7,11,16,22],benefici:16,best:[16,21,22],better:22,between:[3,14,15,19,22],binari:[6,7,20,22],boil:3,boldsymbol:[21,22],boltzmann:[11,19,22],bond:3,bool:[7,8,11,13,16,17],both:[3,11,13],box:[1,3,11,13,14,19,22],box_siz:[11,13,14],build:18,calc_jac:16,calc_p:16,calc_r:16,calcul:[3,6,7,8,12,16,21],call:[3,11,16],can:[0,1,2,3,5,7,13,14,16,19,20,21,22],care:1,carri:[13,15,16],categori:[19,22],chang:[1,3,14],check:[2,3,10,14],check_basin:13,check_warn:14,checkout:7,child:11,choos:11,chosen:[7,16,19,21,22],clear:[19,22],close:[3,19,22],closer:1,closest:[3,13,14],cmd:[],code:[0,2],coeffs_for_each_cv:16,collect:[13,16,21],column:[13,14,16,21,22],colvar:[6,7,8,10,16,20,22],colvar_fil:12,colvar_input:[8,10],colvar_output:12,colvars_fil:16,comb:16,combin:[16,19,20,21,22],command:[0,1,3,5,6,7,12,13],comment:17,commit:[1,3,5,7,11,13,19,22],commitor:1,committor:[1,5,7,13,19,22],common:[1,13],compil:[2,3],complet:[1,3,7,8,11,19,22],compon:14,comput:[19,22],computation:22,concret:13,configur:[19,20,22],consecut:11,consid:[1,5,7,8,11,19,21,22],consist:13,constant:[1,3,16,21,22],constrain:[21,22],construct:[11,13,15,16],contain:[3,5,6,7,11,12,16,19,22],content:9,continu:[5,6,7,11],contribut:[16,21,22],control:[19,22],convent:16,convert:[3,17],convex:[21,22],coord:[0,3,16],coordin:[0,2,3,14,16,17,19,21,22],copi:[0,2,3,11,13,14],copy_out_fil:14,core:[1,2,19,22],correctli:3,correspond:[2,3,10,11,12,13,16,19,21,22],could:[1,19,22],coupl:22,cp2k:[1,3,5,7,10,13],cp2k_dict:14,cp2k_engin:[10,13],cp2k_input:[0,10,13],cp2k_inputs_fil:14,cp2k_output:[10,13],cp2kengin:[3,11,14],cp2kinputshandl:14,cp2koutputhandl:14,creat:[5,7,10,12,13,14],critic:7,csv:[5,6,7,8,10,11,12,16,19,20,21,22],csv_file:[12,16],csv_input:[6,7,8],csv_name:11,cumul:[19,22],cur_index:11,current:[1,5,7,11,13,14],current_offset:11,current_start:11,curv:[16,21,22],cv:[6,7,8,12,16,19,20],d1:1,d:[21,22],dat:[6,7,19,20,22],data:[14,16],datafram:16,debug:[0,2,7],decreas:7,deep:11,defaultacceptor:11,defin:[0,1,2,5,6,7,8,10,11,12,13,14,19,20,21,22],definit:[6,7],delet:[5,7],delta:[0,2,5,7,11,19,22],delta_t:[5,7,13],deltaasd:[],depend:22,descent:16,describ:[0,7,11,22],desir:[2,3,21,22],detail:[1,13,16],determin:[3,11,16],dict:[10,13,14,16,18],dictionari:[3,10,13,14,16,18],did:[13,19,22],differ:[2,5,7,8,11,16,19,20,22],dimens:[13,14,17],direct:[7,11,13,19,22],directli:12,directori:[0,2,5,7,11,13,19,22],disk:15,distanc:1,distribut:[1,11,19,22],diverg:[19,22],divid:[19,22],docstr:14,document:[13,16],doe:[0,13,14,17],don:[6,7,8,21,22],done:[5,7,22],down:3,drawn:[21,22],driver:[9,11,12],dt:[13,15],dump:[6,7],duplic:11,dure:[3,7,8,13,16],dynam:22,e:[2,3,5,7,11,12,16,19,22],each:[0,2,3,7,11,13,14,15,16,19,20,21,22],earli:16,easiest:3,either:[1,2,11,16],element:3,empti:[11,14,16,17,21,22],enabl:16,end:[1,14,17,19,22],engin:[0,2,4,5,7,10,11],engine_dir:[5,7],engine_input:[5,7,10],enough:[2,7,8],ensur:[0,2,3,13],entir:[19,22],entri:[7,20,22],environ:1,eof:[14,17],eoferror:14,equal:1,equival:[1,3,14,21,22],err:[0,2],error:[3,7,10,13,14],essenti:[3,19,22],evalu:[16,21,22],evenli:1,ever:[19,22],everi:[11,14,16],everyth:[3,11],exact:[19,22],exactli:13,exampl:[1,3,7,19,22],except:[0,2,11],exclud:[5,6,7,8],exec:1,execut:[1,3,10,11,13],exist:[0,2,3,11,19,22],exit:[0,2,7,8,19,22],expect:[0,2,3],expens:22,explain:22,explicitli:[6,7,8,10],extens:[5,7],extract:[0,2,3],facilit:[21,22],fail:[0,2,11,19,22],failur:[5,7],fals:[7,8,11,13,14,16],far:22,fast:22,faster:[7,8],fatal:[],featur:1,femtosecond:[5,7,13,14],field:[6,7,8,13],file:[0,1,3,5,6,7,8,10,11,12,13,14,15,16,17,19,20,21,22],file_path:14,filenam:[0,14,15,16,17],find:[16,21,22],finit:[7,8,16],first:[3,13,14,16],fix:[0,16],flag:3,flexibl:[19,22],flip:[13,14,19,22],flip_veloc:[13,14],follow:[1,21,22],form:[0,2,13,14],format:[4,6,10,11,14,16],forward:[1,5,7,13,19,22],found:[0,11,14],frame:[11,13,14,15,16,17],free:3,frequenc:[0,2,3,15],from:[0,1,2,3,6,7,8,10,11,12,13,14,16,17,19,20,21,22],frozenset:16,fs:15,fulfil:3,full:[1,5,7,13,14],func:[21,22],further:[1,19,22],fwd:13,g:[2,3,5,7,11,12],gaussian:[16,21,22],gener:[0,1,2,3,5,7,8,11,12,13,15,16,17,20],generate_veloc:11,genv:1,get:[3,5,7,11,13,14,16],get_atomic_mass_dict:18,get_engine_str:[1,13,14],get_out_fil:14,github:7,give:16,given:[1,3,5,7,8,11,12,13,14,16,17,20,21,22],global:[7,8,10,16,21,22],gmx:3,gmx_mpi:2,go:16,good:[21,22],gradient:16,granular:7,greater:[14,15],gro:[2,3],gro_fil:[2,3],gromac:[1,3,5,7,10,13],gromacsengin:3,grompp:[2,3],grompp_cmd:[2,3],guess:[5,7,11,19,22],ha:[1,3,11,13,14,15,16,17,21,22],had:1,hand:3,handl:[3,13,14,15,17],handler:13,happen:[19,22],have:[3,11,13,14,16,19,22],held:[13,14],here:[5,7,21,22],higher:16,highest:[21,22],hold:[3,11],hop:[21,22],how:[14,15,16,19,22],howev:3,i:[1,5,7,8,11,16,19,21,22],idea:[19,22],ideal:1,ifil:17,ignor:11,immedi:[19,22],immediatlei:[19,22],implement:[1,11,13,14],improv:[7,8,16],includ:[2,3,5,7,13,16,21,22],inclus:16,increas:[16,21,22],independ:[19,22],index:[1,4,11,16],indic:[0,2,16,19,22],individu:[0,2],infer:[6,7,8],info:7,inform:[13,17],infrastructur:3,ing:3,init:[13,15],initi:[2,5,7,11,21],inner:11,inp:3,input:[1,4,5,6,8,10,11,12,13,14,15],input_yml:10,inspect:[5,7],instal:7,instanc:[11,16,19,22],integ:[13,14,15,16,19,22],interact:[3,11],interfac:[12,13],interim:3,intern:3,interpret:22,intersect:11,introduc:11,invalid:1,invok:[1,11,13],io:17,is_accept:[11,16],isn:[13,14],iter:[16,21,22],ith:16,its:[0,1,2,3,6,7,11,13,15,19,22],jacobian:[7,8,16],join:13,just:[1,3,21,22],keep:[11,16,19,22],kei:[16,18],kelvin:[5,7,11,14],keyword:11,kickstart:11,know:11,known:[5,7],kwarg:[11,14],l:[21,22],label:12,later:[10,16,22],latter:[5,7],launch:[5,6,7,13,19,22],lead:[1,11,13],least:[19,22],left:[1,22],length:[0,3,7,8,16,21,22],length_unit:12,less:[16,22],level:[3,7,11,14],like:[0,2,3,19,21,22],likelihood:[7,8,10,19],likelihood_input:10,likelihood_max:10,limit:16,line:[1,7,12,17],linear:[16,21,22],linearli:16,linter:0,list:[0,2,3,5,7,11,13,14,16,17,19,22],ln:[21,22],load:16,local:[7,8,16,21,22],locat:[3,11,13,14,15],log:[3,7,11,21,22],log_level:[0,2],log_nam:11,log_result:11,logger:[11,13,14],longer:16,look:13,loop:11,low:[3,14],m:[3,16,21,22],made:11,mai:[5,7,8,16],main:[7,10],make:[2,11,21,22],mani:[0,2,11,14,16,19,22],manipul:14,mass:[17,18],match:[0,2,13,14,17],mathbf:[16,21,22],matrix:16,matter:11,max:16,max_:[21,22],max_cv:[7,8],max_num_cv:16,maxim:[7,8,16,19],maximizersolut:16,maximum:[7,8,16,21,22],maxwel:[11,19,22],mb:11,md:[3,4,5,7,13,14,15,19,22],md_cmd:[1,2,3,5,7,13],md_input:[6,8,10],mdp:[2,3,10,13],mdp_file:[2,3],mdphandler:15,mdrun:[2,3],mdtraj:3,mean:22,memori:[3,14,15],messag:[7,13,14],metadata:[16,19,22],method:[1,3,11,13,14],minim:[16,21,22],minut:22,miss:[6,7,8],model:[16,21,22],modif:[3,13,15],modifi:[0,2,3,11,13,14,15],modul:[1,4,9],molecular:22,more:[1,11,13,16,19,21,22],most:[16,22],move:[3,11],mpi:1,mpirun:[1,12,13],multibasin:[5,7],multibasinacceptor:11,multipl:[0,2,3,5,7,11,13,14,19,22],multipli:[13,14,16,19,22],must:[3,5,7,11,13,14,15],my_input_fil:1,my_output_fil:1,n:[1,13,14,16,21,22],n_atom:[11,14,17],n_cv:16,n_frame:17,n_iter:[7,8,21,22],n_parallel:[5,7,11],n_point:[5,7,11],n_state_tri:[5,7,11,19,22],n_vel_tri:[5,7,11,19,22],name:[0,1,2,3,5,6,7,8,11,12,13,14,17],nan:[16,21,22],ndarrai:[11,13,14,16,17],necessari:[3,7],necessarili:[0,2],need:[0,1,2,3,6,7,8,19,22],new_loc:[13,14],next:[5,6,7,11,19,21,22],niter:16,node:[5,7],non:[0,2,16,21,22],none:[2,10,11,12,13,14,15,16,17,19,22],normal:1,nostop:1,note:[0,2,3,5,7,11,14,15],noth:2,np:[11,13,14,16],nstxout:15,num:17,num_atom:11,num_cor:1,number:[1,2,3,5,7,8,11,14,15,16,19,20,21,22],numpi:16,o:1,obj:[16,21,22],obj_func:16,obj_val:16,object:[11,12,13,14,15,16,21,22],occur:[0,2,11,12],off:[1,15,16,22],offer:16,offset:[11,13,14],often:[14,15],old:[19,22],onc:[13,14,15,22],one:[1,2,3,5,7,11,13,16,19,20,21,22],onli:[3,5,7,11,16,19,21,22],open:[3,17],optim:[7,8,10],optimum:16,option:[11,13,16],order:[0,7,11,13,14,16,19,22],origin:[0,2,3,13,14,15,19,22],other:[3,5,7,11,12,21,22],otherwis:[11,13,14,16,19,21,22],our:[19,21,22],out:[0,2,5,7,13,14],out_nam:13,outer:11,output:[1,5,7,12,13,14,20],output_nam:[5,6,7,8],over:[5,7,11,13,15,20,22],overal:[19,22],overrid:[3,13],overridden:14,overwrit:[14,15],own:[0,2,19,22],p0:[16,21,22],p:16,p_0:[16,21,22],packag:[4,9,22],page:4,pair:[11,19,20,22],paper:[19,22],parallel:[1,2,5,7,11,13],param:[],paramet:[0,2,3,10,11,12,13,14,15,16,17,19,22],parent:11,pars:[0,3,10,13,14,17],parse_aimless:10,parse_colvar:10,parse_engin:[3,10],parse_likelihood:10,part:[3,5,6,7,22],particular:1,pass:[0,3,11,12,13,14,15],path:[0,1,2,5,6,7,8,10,12,13,14,15,16,17,20,21,22],pbc:[19,22],pd:16,pdb:3,per:[7,8,14,16],perform:[7,8,16,21,22],period:[3,11,13,14,17],periodic_t:10,perturb:[19,22],peter:[7,8,19,22],phase:22,pick:[11,22],pick_start:11,pin:2,pipelin:7,place:[5,7,13,17],plu:[16,21,22],plume:[1,5,6,7,10,11,12,14,16,19,20,22],plumed_cmd:[6,7,12],plumed_driv:10,plumed_fil:[1,5,6,7,12,13],plumed_file_path:14,plumed_handl:13,plumed_out_fil:13,plumeddriv:12,plumedinputhandl:[3,13],plumedoutputhandl:[3,13],po:14,point:[0,1,2,3,5,7,11,13,16,21],posit:[0,2,11,13,14,17],position_dir:11,possibl:[19,22],potenti:22,pr:[16,21,22],preced:[7,8],prefer:3,prefix:[5,7],prepend:1,preprocess:3,present:[0,5,6,7,14,15,16],pressur:1,previou:[6,7,8,11,19,22],primarili:2,principl:[19,22],print:[0,2,3,12,14,15],print_freq:15,probabl:[16,21,22],problem:3,proce:[19,22],process:[1,3,13,19,22],prod_:[21,22],produc:[6,7,8],product:[5,7,11,19,22],program:[7,8,19,22],project:14,projnam:[0,2,3,14],properti:[11,13,14,15],proport:[19,22],provid:[0,20,22],ps:15,purpos:11,put:[7,8],quickli:22,r:[16,21,22],r_jac:16,r_val:16,rais:[11,13,14,15,16,17],random:11,randomli:[11,19,22],raw:[14,15],raw_str:15,re:[3,11],reach:[14,16,17,19,22],reactant:[5,7,11,19,22],reaction:[1,16,19,21,22],read:[3,13,14,15,17],read_and_run:10,read_frames_2_3:14,read_timestep:14,read_xyz_fil:17,read_xyz_fram:17,reader:3,readi:[3,10,17],real:13,receiv:3,recombin:[19,22],recompil:3,record:[11,19,22],rectangular:16,ref:[],refer:[16,19,22],reflect:[0,2],regener:11,reject:[5,7,11,16,19,21,22],rel:1,relev:[0,2,13,14],reli:1,remain:[19,22],repeat:[11,20,21,22],repeatedli:13,repres:[11,13,14,16,21,22],represent:[3,10,12,13,14,16],req_improv:16,request:[3,16],requir:[0,2,5,6,7,8,13,14,16],resampl:[0,2,5,7,11,19,22],reselect:[5,7],resourc:[19,22],respect:[3,11,13,16],restart:[19,22],result:[6,7,8,11,13,16,19,20],resultlogg:11,results_logg:11,resultslogg:11,retri:[5,7],rev:13,revers:[5,7,13,19,22],root:[0,2,11],row:[11,13,14,16,19,21,22],rst:[],run:[0,1,2,5,6,7,8,10,11,12,13,19,22],run_aimless:10,run_arg:11,run_colvar:10,run_likelihood:10,run_shooting_point:13,runner:11,runtimewarn:16,rxn:16,s:[0,2,3,7,13,14,19,21,22],said:[19,22],same:[11,13,14,16,19,22],sampl:[4,7,21],save:[13,14],scale:16,seamlessli:3,search:[4,22],second:[3,14,16],section:[0,3,5,6,7,8,10,13],see:[1,5,7,13,16,19,22],seed:[19,22],select:[11,19,22],separ:11,sequenc:[3,11,13,14,17],sequenti:[19,22],seri:[21,22],serv:14,set:[0,2,3,6,7,8,10,11,13,14,15,16,19],set_delta_t:[3,13,14],set_plumed_fil:14,set_posit:[3,13,14],set_project_nam:14,set_traj_print_fil:14,set_traj_print_freq:[14,15],set_veloc:[3,13,14],shape:[11,13,14,17],share:[1,11],shoot:[0,1,2,3,5,6,7,8,10,11,12,13,16,20,21],shootingresult:[11,13],should:[1,2,3,5,6,7,11,13,14,15,16,19,22],shown:7,signific:16,significantli:[21,22],similar:3,simpli:[20,22],simul:[0,1,2,5,7,11,13,14,15,19,22],simultan:[11,19,22],sinc:[1,19,22],singl:[3,5,7,11,13,16,17,21,22],singlesolut:16,size:[1,11,13,14,17,19,22],slight:3,slightli:[19,22],small:[19,22],so:[0,1,2,3,5,7,13,19,21,22],sol:16,solut:[3,16,21,22],some:[1,16,21,22],someth:[0,2],space:[13,16],spawn:13,specif:[5,7,11,13,14,16,19,22],specifi:[0,1,2,3,13,19,22],speedup:16,split:[7,11],stabl:[19,22],stage:3,standard:[1,14,20,22],start:[0,1,2,3,5,7,11,13,16,21],starts_dir:[5,7],state:[0,1,2,3,5,7,11,14,16,19,22],statist:[16,22],std:[0,2],step:[13,14,15,19,22],still:[3,12,16,19,22],stop:1,store:[13,15,19,21,22],str:[5,6,7,8,10,11,12,13,14,15,16,17,18],string:[1,3,5,7,11,12,13,14,15,17],structur:14,submodul:9,subpackag:9,subprocess:3,subset:[21,22],success:[11,16],suffici:22,sum_:[16,21,22],summari:22,suppli:14,support:1,suppos:1,surfac:[19,22],symbol:[17,18],synchron:11,system:[0,1,2,3,21,22],t:[0,2,5,6,7,8,11,13,14,16,19,21,22],tabl:[3,11,13,14,17],take:[1,3,7,19,21,22],taken:[13,14,19,22],tanh:[16,21,22],temp:[5,7,11,14],temperatur:[5,7,11,14],templat:[3,11,13,14,15],temporari:[5,7,13],term:[16,19,22],termin:[0,2],test:[11,16],text:[21,22],than:[1,11,14,15,19,21,22],thei:[0,1,2,5,7,11,16,19,22],them:[3,11,19,22],therefor:[3,19,22],thi:[0,1,2,3,5,6,7,8,10,11,12,13,14,15,16,19,20,21,22],thing:13,think:3,third:[3,14],those:[0,3,10],three:[2,3,7,22],threshold:16,through:[13,22],thrown:0,ti:13,time:[0,2,5,7,11,13,14,15,19,22],timestep:[3,15],tmp:[5,7],to_csv:16,to_datafram:16,to_opt:16,token:13,told:3,too:16,tool:0,top:[2,3],top_fil:[2,3],topolog:3,total:[5,7,11,19,22],tp:[16,21,22],tpr:[2,3],track:11,traj:14,trajectori:[0,1,2,3,5,7,11,13,14,15,19,22],transit:[0,2,4,7,11,16,19,21],transitori:22,tri:11,trout:[19,22],tupl:[13,14,16,17],turn:[15,16],two:[1,3,5,7,11,13,16,19,22],type:[5,7,10,11,12,13,14,15,16,17,18],unaccept:11,unclear:[19,22],uncommon:[19,22],union:[13,16,17],uniqu:3,unit:[3,12],until:[19,22],up:[1,2,11,19,22],updat:3,upon:[7,19,22],us:[0,1,2,3,4,5,8,10,11,13,14,15,16,19,20,21,22],use_jac:[7,8,16],user:[12,22],usual:16,util:[10,16],valid:[1,3,10,11,13,14],validate_input:[3,13,14],valu:[7,13,14,15,16,18,19,21,22],valueerror:[11,13,14,15,16,17],variabl:[1,10,16,21],ve:[6,7,8,11],vector:16,vel:3,veloc:[0,2,5,7,11,13,14,15,19,22],version:[3,15],via:3,vs:22,wa:[3,11,13,14,16,19,22],wai:3,wait:13,want:[0,6,7,8,21,22],warn:[0,2,3,7,14],we:[1,3,5,6,7,11,19,21,22],weight:[16,21,22],well:[0,2,11,13,14,16,19,22],went:[5,7],were:[1,22],what:[3,5,7,11,13,14,19,22],whatev:3,when:[1,3,13,19,22],where:[0,2,5,7,11,13,16,19,21,22],which:[3,5,7,8,11,13,16,21,22],within:[13,16],without:[11,22],work:[0,2,3,5,7,11,13,22],workflow:[4,7],working_dir:[13,14],would:1,wrapper:[13,16],write:[11,13,14,15,16,17],write_cp2k_input:14,write_mdp:15,write_plum:[3,13],write_xyz_fram:17,written:[3,11,13,14,15],wrong:[0,2,5,7],x:[11,13,14,16,21,22],xyz:[5,6,7,10,11,12,13,14,19,20,22],xyz_fil:12,xyz_input:[6,7],xyz_nam:11,y:[11,13,14],yaml:7,yield:16,yml:[4,10],you:[0,1,2,3,6,7,8,19,22],your:[0,2,19,22],yourself:3,z:[11,13,14],zero:[0,2]},titles:["CP2K Documentation","MD Engines","GROMACS Documentation","Implementing a New Engine","Welcome to transition_sampling\u2019s documentation!","<code class=\"docutils literal notranslate\"><span class=\"pre\">md_inputs</span></code> (Optional)","<code class=\"docutils literal notranslate\"><span class=\"pre\">colvar_inputs</span></code> (Optional)","Using transition_sampling","<code class=\"docutils literal notranslate\"><span class=\"pre\">likelihood_inputs</span></code> (Optional)","transition_sampling","transition_sampling package","transition_sampling.algo package","transition_sampling.colvar package","transition_sampling.engines package","transition_sampling.engines.cp2k package","transition_sampling.engines.gromacs package","transition_sampling.likelihood package","transition_sampling.util package","transition_sampling.util.atomic_masses package","Aimless Shooting","Collective Variable Calculation","Likelihood Maximization","Transition Sampling Workflow"],titleterms:{"2":3,"new":3,"return":3,_launch_traj:3,abstract_engin:13,acceptor:11,addit:[0,2,3],aimless:[19,22],aimless_shoot:11,algo:11,all:[21,22],atom:3,atomic_mass:18,avail:1,basin:[19,22],calcul:[20,22],candid:[21,22],collect:[20,22],colvar:12,colvar_input:[6,7],command:2,content:[4,10,11,12,13,14,15,16,17,18],cp2k:[0,14],cp2k_engin:14,cp2k_input:14,cp2k_output:14,cv:[21,22],defin:3,delta:3,document:[0,2,4],driver:10,engin:[1,3,13,14,15],figur:3,file:2,format:7,frame:3,gener:[19,22],gromac:[2,15],how:3,implement:3,indic:4,individu:[21,22],initi:[19,22],input:[0,2,3,7],integr:3,intro:1,likelihood:[16,21,22],likelihood_input:[7,8],likelihood_max:16,log:[0,2],maxim:[21,22],md:1,md_input:[5,7],mdp:15,modul:[10,11,12,13,14,15,16,17,18],optim:[16,21,22],option:[5,6,7,8],out:3,output:[0,2,3,19,22],packag:[10,11,12,13,14,15,16,17,18],parallel:[19,22],periodic_t:17,plume:[3,13],plumed_driv:12,point:[19,22],posit:3,requir:3,result:[21,22],run:3,s:4,sampl:22,set:[21,22],shoot:[19,22],simul:3,specif:[0,2],start:[19,22],store:3,submodul:[10,11,12,13,14,15,16,17],subpackag:[10,13,17],suggest:3,t:3,tabl:4,templat:2,transit:22,transition_sampl:[4,7,9,10,11,12,13,14,15,16,17,18],us:7,util:[17,18],variabl:[20,22],veloc:3,welcom:4,workflow:[3,22],write:3,xyz:17,yml:7,your:3}})