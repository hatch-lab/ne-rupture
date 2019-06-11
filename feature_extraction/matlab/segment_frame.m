% for IC200 20x 1:4 sized image (640x540), vse=3, vse2=3, bwao=10
% for 60x DV full size (D,10,8,250);

% Copied from code provided by Julien
function [statsNUC, Lnuc]=segment_frame(DAPI,vse,vse2,bwao,yn)
  % This is the primary function I use to segment nuclei based on DAPI
  % staining. It is inspired from the marker-controlled watershed
  % segmentation example from the Mathworks documentation.
  % https://www.mathworks.com/help/images/marker-controlled-watershed-segmentation.html
  %
  % Input arguments:
  % DAPI     uint16 grayscale DAPI image
  % vse      radius of the disk structuring element used in the opening
  % reconstruction step (range from 4 to 20)
  % vse2     radius of the disk structuring element used in the foreground
  % detection step (range from 0 to 10)
  % bwao     object size to be removed (range from 25 to 300)
  % yn       logical argument that specifies whether subsegmentation shall be
  % applied using the Nuclear2 function (0- No, 1- yes)
  %
  % Hints:
  % Preprocessing of the DAPI image is key for successfull segmentation.
  [a,b]=size(DAPI); %J is already adjusted here

  %h=fspecial('average',1);
  %h=fspecial('log',8,5);
  %Df=imfilter(DAPI,h);
  Df=DAPI;
  % mode type background subtraction

  % k1=max(DAPI(:))/2;
  % k2=DAPI(:)<=k1;
  % km=mode(DAPI(k2));
  % 
  % Dfb=DAPI-2*km;

  % k1=max(Df(:))/2;
  % k2=Df(:)<=k1;
  % km=mode(Df(k2));
  % 
  % Dfb=Df-1*km;

  %Dfb=imtophat(Dfb,strel('disk',100));
  % Dfb=imadjust(Dfb);
  % Gr=reshape(GFP,numel(GFP),1);
  % q=quantile(Gr,0.05);
  % Gm=GFP-q;
  Dfb=Df;
  % create a first gradient based on Sobel edge detection
  hy=fspecial('sobel');
  hx=hy';
  Iy=imfilter(double(Dfb),hy,'replicate');
  Ix=imfilter(double(Dfb),hx,'replicate');
  gradmag=sqrt(Ix.^2+Iy.^2);


  % define markers by image opening and reconstruction
  se=strel('disk',vse);
  Jo=imopen(Dfb,se); 
  Je=imerode(Jo,se);
  Jor=imreconstruct(Je,Dfb);


  %foreground detection
  fgm=imregionalmax(Jor,4);

  %se2=strel(ones(vse2,vse2));
  se2=strel('disk',vse2);
  fgm2=imclose(fgm,se2);
  fgm3=imerode(fgm2,se2);
  fgm4=bwareaopen(fgm3,bwao);


  % Local background delimitation
  bw=imbinarize(Jor,graythresh(Jor));
  %bw=imbinarize(Jor,'adaptive');figure, imshow(bw);
  %bw=imclearborder(bw);
  D=bwdist(bw);
  DL=watershed(D);
  bgm=DL==0;


  %apply combined gradient
  gradmag2=imimposemin(gradmag,bgm | fgm4);

  L=watershed(gradmag2);

  rem=mode(L(:));
  Ll=L;
  Ll=imclearborder(Ll);
  Ll(L==rem)=0;


  %resegment with regular watershed to break down fused objects
  if yn
    [~,~, Lnuc]=Nuclear2(Ll,0);
    %[~,~, Lnuc]=Nuclear(Ll);
    Lnuc=bwmorph(Lnuc,'thicken',2);
  else
    Lnuc=Ll; % to use when second segmentation step is not needed
  end


  statsNUC=regionprops(Lnuc,DAPI,'Area','PixelIdxList','PixelValues','MeanIntensity','BoundingBox','Solidity');

  %% filter nuclei

  %       A=[statsNUC.Area];
  %      I=[statsNUC.MeanIntensity];
  %      tf=A>600  | I<2000;
  % 
  %      statsNUC(tf)=[];

  MASK=false(a,b);
  Pix=vertcat(statsNUC.PixelIdxList);
  MASK(Pix)=true;
  Lnuc=bwlabel(MASK);
  %% view segmentation results (uncomment next three lines)
  %    Lrgb=label2rgb(Ll,'jet','k','shuffle');
  %   figure, imshow(DAPI,[]);hold on;
  %  himage=imshow(Lrgb);set(himage,'AlphaData',0.2);%title(filename)
   
  Lnuc=Lnuc>0;
