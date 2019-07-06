function AllStatsPath=process_video(FramePaths, PxSize, OutputPath, MaskPath)

  NumFrames = numel(FramePaths);

  AllStats=cell(0,18);

  parpool('local')
  parfor i=1:NumFrames
    I=imread(FramePaths{i}); %read original image from ith frame
    Stats=get_frame_features(I,i,PxSize,MaskPath);

    % Filter by size and intensity
    % TMI=cell2mat(Stats(:,13)); % Nuc Mean
    % TS=cell2mat(Stats(:,11)); % Nuc Area
    % tf=TMI<45 | TS<900 | TS>6000;
    % Stats(tf,:)=[];

    % Delete nuclei that are too close to others
    % C=reshape([cell2mat(Stats(:,7:8))],2,numel(Stats(:,1)))';
    % pd=squareform(pdist(C));
    % [r c]=find(pd<50 & pd>0);
    % Stats(unique(r),:)=[];
    AllStats=[ AllStats ; Stats ];
    disp(i)
  end

  Table = cell2table(AllStats,'VariableNames',{ 'particle_id', 'frame', 'x', 'y', 'weighted_x_cyto', 'weighted_y_cyto', 'x_nuc', 'y_nuc', 'x_cyto', 'y_cyto', 'area_nuc', 'area_cyto', 'mean_nuc', 'mean_cyto', 'min_nuc', 'min_cyto', 'max_nuc', 'max_cyto' });
  writetable(Table,OutputPath);
  AllStatsPath = OutputPath;




