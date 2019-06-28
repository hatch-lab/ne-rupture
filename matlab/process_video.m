function AllStatsPath=process_video(FramePaths, MFramePaths, PxSize, OutputPath, MaskPath)

  NumFrames = numel(FramePaths);

  AllStats=cell(0,26);

  parpool('local')
  parfor i=1:NumFrames
    I=imread(MFramePaths{i}); %read processed image from ith frame
    O=imread(FramePaths{i}); %read original image from ith frame
    Stats=get_frame_features(I,O,i,PxSize,MaskPath);
    AllStats=[ AllStats ; Stats ];
    disp(i)
  end

  Table = cell2table(AllStats,'VariableNames',{ 'particle_id', 'frame', 'x', 'y', 'weighted_x_raw_nuc', 'weighted_y_raw_nuc', 'weighted_x_raw_cyto', 'weighted_y_raw_cyto', 'weighted_x_proc_nuc', 'weighted_y_proc_nuc', 'weighted_x_proc_cyto', 'weighted_y_proc_cyto', 'area_nuc', 'area_cyto', 'mean_raw_nuc', 'mean_raw_cyto', 'mean_proc_nuc', 'mean_proc_cyto', 'min_raw_nuc', 'min_raw_cyto', 'min_proc_nuc', 'min_proc_cyto', 'max_raw_nuc', 'max_raw_cyto', 'max_proc_nuc', 'max_proc_cyto' });
  writetable(Table,OutputPath);
  AllStatsPath = OutputPath;




